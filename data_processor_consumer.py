import json
import logging
from pathlib import Path
import signal
import os

import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_processor_consumer.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DataProcessorConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        input_topic: str,
        output_topic: str,
        group_id: str = "data-processor-consumer-group",
        batch_size: int = 1000,
    ):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.batch_size = batch_size
        self.transactions_buffer = []
        self.processed_count = 0
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

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        Path("logs").mkdir(parents=True, exist_ok=True)

        logger.info("Kafka Data Processor Consumer initialized")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        logger.info(f"Input Topic: {input_topic}")
        logger.info(f"Output Topic: {output_topic}")
        logger.info(f"Group ID: {group_id}")
        logger.info(f"Batch size: {batch_size}")

    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()

        # Time-based features
        df_processed["hour_of_day"] = (df_processed["Time"] % (24 * 3600)) / 3600
        df_processed["hour_sin"] = np.sin(2 * np.pi * df_processed["hour_of_day"] / 24)
        df_processed["hour_cos"] = np.cos(2 * np.pi * df_processed["hour_of_day"] / 24)
        df_processed["is_night"] = (
            (df_processed["hour_of_day"] >= 22) | (df_processed["hour_of_day"] <= 6)
        ).astype(int)

        # Amount features
        df_processed["Amount_log"] = np.log1p(df_processed["Amount"])
        df_processed["is_high_amount"] = (df_processed["Amount"] > 200).astype(int)

        # V-columns features
        v_cols = [c for c in df_processed.columns if c.startswith("V")]
        if v_cols:
            df_processed["V_mean"] = df_processed[v_cols].mean(axis=1)
            df_processed["V_std"] = df_processed[v_cols].std(axis=1)
            df_processed["V_outlier_count"] = (df_processed[v_cols].abs() > 3).sum(
                axis=1
            )

        # Fraud label
        if "Class" in df_processed.columns:
            try:
                df_processed["is_fraud"] = (
                    df_processed["Class"]
                    .astype(str)
                    .str.replace('"', "")
                    .astype(float)
                    .astype(int)
                )
            except (ValueError, TypeError):
                logger.warning("Could not convert Class to is_fraud, defaulting to 0")
                df_processed["is_fraud"] = 0
        else:
            df_processed["is_fraud"] = 0

        logger.debug(f"Preprocessed batch of {len(df_processed)} transactions")

        return df_processed

    def send_processed_batch(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            transaction = row.to_dict()

            key = str(transaction.get("transaction_id", ""))
            try:
                self.producer.send(self.output_topic, key=key, value=transaction)
            except Exception as e:
                logger.error(f"Failed to send processed transaction: {e}")

        self.producer.flush()
        logger.info(f"Sent {len(df)} processed transactions to {self.output_topic}")

    def process_message(self, message):
        try:
            transaction = message.value
            self.transactions_buffer.append(transaction)

            if len(self.transactions_buffer) >= self.batch_size:
                self.process_buffer()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def process_buffer(self):
        if not self.transactions_buffer:
            return

        try:
            df = pd.DataFrame(self.transactions_buffer)

            df_processed = self.preprocess_batch(df)

            self.send_processed_batch(df_processed)

            self.processed_count += len(self.transactions_buffer)
            logger.info(
                f"Processed batch of {len(self.transactions_buffer)} transactions. Total: {self.processed_count}"
            )

            self.transactions_buffer = []

        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            self.transactions_buffer = []

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

        if self.transactions_buffer:
            logger.info(
                f"Processing remaining {len(self.transactions_buffer)} transactions..."
            )
            self.process_buffer()

        self.consumer.close()
        self.producer.close()

        logger.info(
            f"Consumer closed. Total transactions processed: {self.processed_count}"
        )


def main():
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9094,localhost:9096,localhost:9098"
    )
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "raw-data")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "processed-data")
    batch_size = int(os.getenv("BATCH_SIZE", "1000"))

    consumer = DataProcessorConsumer(
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic,
        output_topic=output_topic,
        batch_size=batch_size,
    )

    consumer.start()


if __name__ == "__main__":
    main()
