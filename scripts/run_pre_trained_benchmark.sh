#!/bin/bash

# Usage: ./run_pre_trained_benchmark.sh <model_name>
# Example: ./run_pre_trained_benchmark.sh Qwen/Qwen2.5-3B-Instruct

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME="$1"
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '-')

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
    --batch-size 1024

  echo "Finished dataset: $DATASET"
done