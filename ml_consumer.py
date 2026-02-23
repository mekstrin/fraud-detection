import json
import logging
from pathlib import Path
from datetime import datetime
import signal
import os
import pickle

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/ml_consumer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MLConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        input_topic: str,
        output_topic: str,
        group_id: str = "ml-consumer-group",
        model_path: str = "model/saved/fraud_detector.pkl",
    ):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.model_path = model_path
        self.running = True

        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_records=500,
        )

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )

        self.load_model()

        Path("logs").mkdir(parents=True, exist_ok=True)

        logger.info("Kafka ML Consumer initialized")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        logger.info(f"Input Topic: {input_topic}")
        logger.info(f"Output Topic: {output_topic}")
        logger.info(f"Group ID: {group_id}")
        logger.info(f"Model path: {model_path}")

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data.get("feature_names", [])
            logger.info(
                f"Model loaded successfully. Expecting {len(self.feature_names)} features."
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.feature_names = []

    def prepare_features(self, transaction: dict):
        df = pd.DataFrame([transaction])

        for col in self.feature_names:
            if col not in df.columns:
                logger.warning(f"Feature {col} missing in input, filling with 0")
                df[col] = 0

        X = df[self.feature_names]

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        return X_scaled

    def predict(self, transaction: dict) -> dict:
        if self.model is None:
            return {
                "transaction_id": transaction.get("transaction_id"),
                "prediction": None,
                "error": "Model not loaded",
            }

        try:
            X = self.prepare_features(transaction)
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]

            # Merge prediction with original transaction data
            result = transaction.copy()
            result.update(
                {
                    "model_prediction": int(prediction),
                    "fraud_probability": float(probability),
                    "ml_processed_at": datetime.now().isoformat(),
                }
            )
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "transaction_id": transaction.get("transaction_id"),
                "prediction": None,
                "error": str(e),
            }

    def send_ml_result(self, result: dict):
        try:
            key = str(result.get("transaction_id", ""))
            self.producer.send(self.output_topic, key=key, value=result)
            logger.debug(
                f"Sent ML result for transaction {result.get('transaction_id')}"
            )
        except Exception as e:
            logger.error(f"Failed to send ML result: {e}")

    def process_message(self, message):
        try:
            transaction = message.value

            result = self.predict(transaction)

            self.send_ml_result(result)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def start(self):
        logger.info("Starting to consume messages...")
        logger.info("Press Ctrl+C to stop")

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            for message in self.consumer:
                if not self.running:
                    break

                self.process_message(message)

        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        finally:
            self.close()

    def signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.running = False

    def close(self):
        logger.info("Closing consumer...")

        self.consumer.close()
        self.producer.close()

        logger.info("Consumer closed.")


def main():
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9094,localhost:9096,localhost:9098"
    )
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "processed-data")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "ml-results")
    model_path = os.getenv("MODEL_PATH", "model/saved/fraud_detector.pkl")

    consumer = MLConsumer(
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic,
        output_topic=output_topic,
        model_path=model_path,
    )

    consumer.start()


if __name__ == "__main__":
    main()
