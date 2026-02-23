import json
import time
import argparse
import logging
from pathlib import Path
from typing import Literal
import random
import uuid

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/producer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TransactionProducer:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        delay_mode: Literal["static", "random"] = "static",
        delay_static: float = 0.01,
        delay_min: float = 0.001,
        delay_max: float = 0.01,
        checkpoint_file: str = "logs/producer_checkpoint.txt",
    ):
        self.topic = topic
        self.delay_mode = delay_mode
        self.delay_static = delay_static
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.checkpoint_file = checkpoint_file

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            max_in_flight_requests_per_connection=1,
        )

        logger.info("Kafka Producer initialized")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Delay mode: {delay_mode}")

    def _load_checkpoint(self) -> int:
        checkpoint_path = Path(self.checkpoint_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    idx = int(f.read().strip())
                    logger.info(f"Loaded checkpoint: resuming from index {idx}")
                    return idx
            except (ValueError, IOError) as e:
                logger.warning(
                    f"Failed to load checkpoint: {e}. Starting from beginning."
                )
                return 0
        return 0

    def _save_checkpoint(self, idx: int):
        try:
            checkpoint_path = Path(self.checkpoint_file)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                f.write(str(idx))
        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _clear_checkpoint(self):
        try:
            checkpoint_path = Path(self.checkpoint_file)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint cleared after successful completion")
        except IOError as e:
            logger.error(f"Failed to clear checkpoint: {e}")

    def send_transaction(self, transaction: dict) -> bool:
        try:
            key = str(transaction.get("transaction_id", ""))
            self.producer.send(self.topic, key=key, value=transaction)
            return True

        except KafkaError as e:
            logger.error(f"Failed to send transaction: {e}")
            return False

    def stream_from_dataframe(self, df: pd.DataFrame):
        start_idx = self._load_checkpoint()
        original_total = len(df)

        if start_idx > 0:
            logger.info(
                f"Resuming from index {start_idx} (skipping {start_idx} already processed transactions)"
            )
            df = df.iloc[start_idx:].reset_index(drop=True)

        total = len(df)
        sent = 0
        failed = 0
        start_time = time.time()

        logger.info(
            f"Starting to stream {total} transactions (total in dataset: {original_total})..."
        )
        logger.info(f"Delay mode: {self.delay_mode}")

        current_absolute_idx = start_idx

        try:
            for idx, row in df.iterrows():
                transaction = row.to_dict()

                if "transaction_id" not in transaction:
                    transaction["transaction_id"] = str(uuid.uuid4())

                if self.send_transaction(transaction):
                    sent += 1
                else:
                    failed += 1

                current_absolute_idx = start_idx + idx + 1

                if sent % 100 == 0:
                    self._save_checkpoint(current_absolute_idx)

                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = sent / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {current_absolute_idx}/{original_total} | "
                        f"Sent: {sent} | Failed: {failed} | "
                        f"Rate: {rate:.2f} msgs/sec"
                    )

                if self.delay_mode == "static":
                    time.sleep(self.delay_static)
                else:
                    delay = random.uniform(self.delay_min, self.delay_max)
                    time.sleep(delay)

            elapsed = time.time() - start_time
            rate = sent / elapsed if elapsed > 0 else 0

            logger.info("=" * 60)
            logger.info("Streaming completed!")
            logger.info(f"Total transactions in dataset: {original_total}")
            logger.info(f"Successfully sent: {sent}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            logger.info(f"Average rate: {rate:.2f} messages/second")
            logger.info("=" * 60)

            self._clear_checkpoint()

        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
            self._save_checkpoint(current_absolute_idx)
            logger.info(f"Checkpoint saved at index {current_absolute_idx}")
        finally:
            self.close()

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed")


def rebalance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    is_fraud = df["Class"].astype(str) == "1"
    frauds = df[is_fraud]
    normals = df[~is_fraud]

    if len(frauds) == 0:
        logger.warning("No fraud transactions found to rebalance.")
        return df

    logger.info(f"Rebalancing: {len(frauds)} frauds, {len(normals)} normals")

    if len(frauds) > 0:
        interval = int(len(normals) / len(frauds))
    else:
        interval = 0

    if interval == 0:
        interval = 1

    logger.info(f"Injecting fraud every ~{interval} transactions")

    reordered = []
    f_idx = 0
    n_idx = 0

    frauds_list = frauds.to_dict("records")
    normals_list = normals.to_dict("records")

    while n_idx < len(normals_list):
        end_n = min(n_idx + interval, len(normals_list))
        reordered.extend(normals_list[n_idx:end_n])
        n_idx = end_n

        if f_idx < len(frauds_list):
            reordered.append(frauds_list[f_idx])
            f_idx += 1

    if f_idx < len(frauds_list):
        reordered.extend(frauds_list[f_idx:])

    return pd.DataFrame(reordered)


def main():
    parser = argparse.ArgumentParser(description="Kafka Transaction Producer")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "random"],
        default="static",
        help="Delay mode: static or random",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/transactions.csv",
        help="Path to transaction dataset",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of transactions to send"
    )
    parser.add_argument(
        "--uniform-fraud",
        action="store_true",
        help="Distribute fraud transactions uniformly",
    )

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9094,localhost:9096,localhost:9098"
    )
    topic = os.getenv("KAFKA_TOPIC", "raw-data")
    delay_min = float(os.getenv("PRODUCER_SLEEP_MIN", "0.001"))
    delay_max = float(os.getenv("PRODUCER_SLEEP_MAX", "0.01"))

    dataset_path = Path(args.dataset)
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {args.limit} transactions")

    logger.info(f"Loaded {len(df)} transactions")

    if args.uniform_fraud:
        df = rebalance_dataset(df)
        logger.info("Dataset rebalanced for uniform fraud distribution")

    producer = TransactionProducer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        delay_mode=args.mode,
        delay_min=delay_min,
        delay_max=delay_max,
    )

    producer.stream_from_dataframe(df)


if __name__ == "__main__":
    main()
