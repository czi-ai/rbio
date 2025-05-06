#!/bin/bash

# Usage: ./run_pre_trained_benchmark.sh <model_name> [batch_size]
# Example: ./run_pre_trained_benchmark.sh Qwen/Qwen2.5-3B-Instruct 512

set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_name> [batch_size]"
  exit 1
fi

MODEL_NAME="$1"
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '-')
BATCH_SIZE="${2:-1024}"  # Default to 1024 if not provided

# Dataset list
DATASETS=("rpe1" "jurkat" "k562" "hepg2")

for DATASET in "${DATASETS[@]}"; do
  echo "Running pre-trained benchmark for dataset: $DATASET"

  INPUT_PATH="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/${DATASET}-test-v0.1.1-no-augmentation.csv"
  OUTPUT_PATH="/mnt/czi-sci-ai/project-rbio/benchmarks/${MODEL_NAME_SAFE}-STOCK.stats.${DATASET}.csv"

  python -m czi.ai.rbio.benchmarks.benchmark_pre_trained \
    --dataset-path "$INPUT_PATH" \
    --model-name "$MODEL_NAME" \
    --output-path "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE"

  echo "Finished dataset: $DATASET"
done