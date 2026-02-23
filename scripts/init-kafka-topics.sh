#!/bin/bash

set -e

echo "Creating topic: raw-data"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 \
  --create \
  --if-not-exists \
  --topic raw-data \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --config min.insync.replicas=2

echo "Creating topic: processed-data"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 \
  --create \
  --if-not-exists \
  --topic processed-data \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --config min.insync.replicas=2

echo "Creating topic: visualization"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 \
  --create \
  --if-not-exists \
  --topic visualization \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --config min.insync.replicas=2

echo "Creating topic: ml-results"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 \
  --create \
  --if-not-exists \
  --topic ml-results \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config segment.ms=86400000 \
  --config min.insync.replicas=2

echo "Listing all topics:"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 --list

echo "Topic details:"
kafka-topics.sh --bootstrap-server kafka-1:9092,kafka-2:9092,kafka-3:9092 --describe --topic raw-data

echo "Kafka topics initialization completed successfully!"