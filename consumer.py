import json
import logging
from pathlib import Path
from datetime import datetime
import signal
import os

import pandas as pd
import numpy as np
from kafka import KafkaConsumer
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/consumer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TransactionConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str = "transaction-consumer-group",
        batch_size: int = 1000,
    ):
        self.topic = topic
        self.batch_size = batch_size
        self.transactions_buffer = []
        self.processed_count = 0
        self.running = True

        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_records=500,
        )

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        logger.info("Kafka Consumer initialized")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Group ID: {group_id}")
        logger.info(f"Batch size: {batch_size}")

    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()

        if "Time" in df_processed.columns:
            df_processed["hour_of_day"] = (df_processed["Time"] % (24 * 3600)) / 3600

            df_processed["hour_sin"] = np.sin(
                2 * np.pi * df_processed["hour_of_day"] / 24
            )
            df_processed["hour_cos"] = np.cos(
                2 * np.pi * df_processed["hour_of_day"] / 24
            )

            df_processed["is_night"] = (
                (df_processed["hour_of_day"] >= 22) | (df_processed["hour_of_day"] <= 6)
            ).astype(int)

        if "Amount" in df_processed.columns:
            df_processed["Amount_log"] = np.log1p(df_processed["Amount"])
            df_processed["is_high_amount"] = (df_processed["Amount"] > 200).astype(int)

        v_cols = [c for c in df_processed.columns if c.startswith("V")]
        if v_cols:
            df_processed["V_mean"] = df_processed[v_cols].mean(axis=1)
            df_processed["V_std"] = df_processed[v_cols].std(axis=1)
            df_processed["V_outlier_count"] = (df_processed[v_cols].abs() > 3).sum(
                axis=1
            )

        logger.debug(f"Preprocessed batch of {len(df_processed)} transactions")

        return df_processed

    def save_batch(self, df: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/processed/batch_{timestamp}.parquet")

        df.to_parquet(output_path, index=False)
        logger.info(f"Saved batch to {output_path}")

        cumulative_path = Path("data/processed/all_transactions.csv")

        if cumulative_path.exists():
            try:
                existing_cols = pd.read_csv(cumulative_path, nrows=0).columns
                if len(existing_cols) != len(df.columns):
                    logger.warning(
                        f"Schema mismatch! Existing: {len(existing_cols)}, New: {len(df.columns)}. Rotating file."
                    )
                    backup_path = Path(
                        f"data/processed/all_transactions_backup_{timestamp}.csv"
                    )
                    cumulative_path.rename(backup_path)
                    df.to_csv(cumulative_path, index=False)
                else:
                    df.to_csv(cumulative_path, mode="a", header=False, index=False)
            except Exception as e:
                logger.error(f"Error checking/writing CSV: {e}")
                df.to_csv(cumulative_path, mode="a", header=False, index=False)
        else:
            df.to_csv(cumulative_path, index=False)

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

            self.save_batch(df_processed)

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

        logger.info(
            f"Consumer closed. Total transactions processed: {self.processed_count}"
        )


def main():
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9094,localhost:9096,localhost:9098"
    )
    topic = os.getenv("KAFKA_TOPIC", "raw-data")
    batch_size = int(os.getenv("BATCH_SIZE", "1000"))

    consumer = TransactionConsumer(
        bootstrap_servers=bootstrap_servers, topic=topic, batch_size=batch_size
    )

    consumer.start()


if __name__ == "__main__":
    main()
