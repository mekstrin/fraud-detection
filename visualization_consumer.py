import json
import logging
from pathlib import Path
from datetime import datetime
import signal
import os

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/visualization_consumer.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class VisualizationConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        input_topic: str,
        output_topic: str,
        group_id: str = "visualization-consumer-group",
        batch_size: int = 1000,
    ):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.batch_size = batch_size
        self.transactions_buffer = []
        self.running = True

        self.state_file = Path("data/processed/dashboard_stats.json")
        self.state = self.load_state()

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

        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        logger.info("Kafka Visualization Consumer initialized")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        logger.info(f"State file: {self.state_file}")

    def load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

        return {
            "last_updated": datetime.now().isoformat(),
            "metrics": {
                "total_transactions": 0,
                "fraud_detected": 0,
                "total_amount": 0.0,
                "fraud_amount": 0.0,
                "avg_fraud_probability": 0.0,
                "high_risk_alerts": 0,
            },
            "category_stats": {},
            "recent_transactions": [],
            "sampled_data": [],
        }

    def save_state(self):
        try:
            self.state["last_updated"] = datetime.now().isoformat()

            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.state, f, indent=2)
            temp_file.replace(self.state_file)

            logger.info(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update_state(self, df: pd.DataFrame):
        self.state["metrics"]["total_transactions"] += len(df)
        self.state["metrics"]["total_amount"] += float(df["Amount"].sum())

        # ML-specific metrics
        new_avg_prob = float(df["fraud_probability"].mean())
        current_total = self.state["metrics"]["total_transactions"]
        old_total = current_total - len(df)
        current_avg = self.state["metrics"].get("avg_fraud_probability", 0.0)

        if current_total > 0:
            self.state["metrics"]["avg_fraud_probability"] = (
                (current_avg * old_total) + (new_avg_prob * len(df))
            ) / current_total

        high_risk_count = int((df["fraud_probability"] > 0.8).sum())
        self.state["metrics"]["high_risk_alerts"] = (
            self.state["metrics"].get("high_risk_alerts", 0) + high_risk_count
        )

        fraud_df = df[df["is_fraud"] == 1]
        fraud_count = len(fraud_df)
        fraud_amount = float(fraud_df["Amount"].sum()) if not fraud_df.empty else 0.0

        self.state["metrics"]["fraud_detected"] += fraud_count
        self.state["metrics"]["fraud_amount"] += fraud_amount

        if "merchant_category" in df.columns:
            cat_groups = df.groupby("merchant_category")
            for cat, group in cat_groups:
                stats = self.state["category_stats"].get(
                    str(cat), {"total": 0, "fraud": 0, "avg_prob": 0.0}
                )
                old_total = stats["total"]
                stats["total"] += len(group)
                stats["fraud"] += int(group["is_fraud"].sum())
                group_avg = float(group["fraud_probability"].mean())
                stats["avg_prob"] = (
                    (stats.get("avg_prob", 0.0) * old_total) + (group_avg * len(group))
                ) / stats["total"]

                self.state["category_stats"][str(cat)] = stats

        recent_records = df.tail(20).to_dict(orient="records")

        self.state["recent_transactions"].extend(recent_records)
        self.state["recent_transactions"] = self.state["recent_transactions"][-20:]

        cols_to_keep = [
            "Amount",
            "is_fraud",
            "hour_of_day",
            "location_distance",
            "fraud_probability",
            "model_prediction",
        ]
        existing_cols = [c for c in cols_to_keep if c in df.columns]

        if existing_cols:
            samples = df[existing_cols].tail(1000).to_dict(orient="records")
            self.state["sampled_data"].extend(samples)
            self.state["sampled_data"] = self.state["sampled_data"][-1000:]

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

            self.update_state(df)
            self.save_state()

            self.producer.send(
                self.output_topic,
                value={"status": "updated"},
            )

            logger.info(
                f"Processed batch of {len(self.transactions_buffer)}. Global total: {self.state['metrics']['total_transactions']}"
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

        logger.info("Consumer closed.")


def main():
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9094,localhost:9096,localhost:9098"
    )
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "ml-results")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "visualization")
    batch_size = int(os.getenv("BATCH_SIZE", "50"))

    consumer = VisualizationConsumer(
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic,
        output_topic=output_topic,
        batch_size=batch_size,
    )

    consumer.start()


if __name__ == "__main__":
    main()
