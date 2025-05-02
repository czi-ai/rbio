#!/bin/bash

# Usage: ./run_grpo_benchmark.sh <checkpoint_folder> <model_name>
# Example: ./run_grpo_benchmark.sh /mnt/czi-sci-ai/project-rbio/checkpoints/PertQA-DE/K562/1_Rewrite/Qwen25-3b-Instruct-NORA-AllData/checkpoint-40000 Qwen/Qwen2.5-3B-Instruct

set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <checkpoint_folder> <model_name>"
  exit 1
fi

CHECKPOINT_PATH="$1"
MODEL_NAME="$2"

# Extract step number from checkpoint path, e.g. 'checkpoint-40000' -> 40000
STEP_NUMBER=$(basename "$CHECKPOINT_PATH" | grep -oP '\d+')

if [ -z "$STEP_NUMBER" ]; then
  echo "Could not extract step number from checkpoint path: $CHECKPOINT_PATH"
  exit 1
fi

# Normalize model name for filename use (e.g., replace slashes with dashes)
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '-')

# Dataset list
DATASETS=("rpe1" "jurkat" "k562" "hepg2")

for DATASET in "${DATASETS[@]}"; do
  echo "Running benchmark for dataset: $DATASET"

  INPUT_PATH="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/${DATASET}-test-v0.1.1-no-augmentation.csv"
  OUTPUT_PATH="/mnt/czi-sci-ai/project-rbio/benchmarks/${MODEL_NAME_SAFE}-NORA-NOLEN-AllData-${STEP_NUMBER}.stats.${DATASET}.csv"

  python -m czi.ai.rbio.benchmarks.benchmark_grpo_trained \
    --dataset-path "$INPUT_PATH" \
    --model-name "$MODEL_NAME" \
    --grpo-checkpoint "$CHECKPOINT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --batch-size 1024

  echo "Finished dataset: $DATASET"
done