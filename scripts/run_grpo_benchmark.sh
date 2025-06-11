#!/bin/bash

# Usage: ./run_grpo_benchmark.sh <checkpoint_folder> <model_name> [batch_size]
# Example: ./run_grpo_benchmark.sh /mnt/.../checkpoint-40000 Qwen/Qwen2.5-3B-Instruct 512

set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <checkpoint_folder> <model_name> [batch_size]"
  exit 1
fi

CHECKPOINT_PATH="$1"
MODEL_NAME="$2"
BATCH_SIZE="${3:-1024}"  # Default to 1024 if not provided

# Normalize model name for filename use (e.g., replace slashes with dashes)
# Get the model folder name from the checkpoint path (strip checkpoint-xxxxx and get parent dir)
MODEL_FOLDER_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")

# Extract step number from checkpoint path, e.g. 'checkpoint-40000' -> 40000
STEP_NUMBER=$(basename "$CHECKPOINT_PATH" | grep -oP '\d+')

if [ -z "$STEP_NUMBER" ]; then
  echo "Could not extract step number from checkpoint path: $CHECKPOINT_PATH"
  exit 1
fi


# Extract data subset name from the path (e.g., "All_Data" from ".../PertQA-DE/All_Data/...")
DATA_SUBSET=$(echo "$CHECKPOINT_PATH" | sed -n 's|.*/PertQA-DE/\([^/]*\)/.*|\1|p')

# Dataset list
DATASETS=("rpe1" "jurkat" "k562" "hepg2")

for DATASET in "${DATASETS[@]}"; do
  echo "Running benchmark for dataset: $DATASET"

  INPUT_PATH="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/${DATASET}-test-v0.1.1-no-augmentation.csv"
  OUTPUT_PATH="/mnt/czi-sci-ai/project-rbio/benchmarks/${MODEL_FOLDER_NAME}-${DATA_SUBSET}-${STEP_NUMBER}.stats.${DATASET}.csv"

  python -m czi.ai.rbio.benchmarks.benchmark_grpo_trained \
    --dataset-path "$INPUT_PATH" \
    --model-name "$MODEL_NAME" \
    --grpo-checkpoint "$CHECKPOINT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE"

  echo "Finished dataset: $DATASET"
done