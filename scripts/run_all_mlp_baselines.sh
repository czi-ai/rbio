#!/bin/bash

# Define datasets and embedding types
datasets=(k562 hepg2 jurkat rpe1)
embedding_types=("1_hot" "gene2vec" "esm_emb")

# Paths
DATASET_BASE="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE"
OUTPUT_BASE="/mnt/czi-sci-ai/project-rbio/baselines"
CHECKPOINT_BASE="/mnt/czi-sci-ai/project-rbio/checkpoints/MLP"

# Embedding file paths
EMBEDDING_PATHS=(
    "/mnt/czi-sci-ai/project-rbio/repr/1_hot_embeddings_filled.pkl"
    "/mnt/czi-sci-ai/project-rbio/repr/gene2vec_embeddings_filled.pkl"
    "/mnt/czi-sci-ai/project-rbio/repr/esm_embedding_dictionary_filled.pkl"
)

# First, train models for each training dataset and embedding type
echo "Training phase..."
for train in "${datasets[@]}"; do
  for i in "${!embedding_types[@]}"; do
    embedding_type="${embedding_types[$i]}"
    embedding_file="${EMBEDDING_PATHS[$i]}"
    
    train_file="${DATASET_BASE}/${train}-train-v0.2.0-no-augmentation.csv"
    checkpoint_dir="${CHECKPOINT_BASE}/MLP-${train}-${embedding_type}"

    # Skip if model already exists
    if [ -f "${checkpoint_dir}/mlp_model.pt" ]; then
      echo "Model already exists for train=${train}, embedding=${embedding_type}, skipping training..."
      continue
    fi

    echo "Training model for: train=${train}, embedding=${embedding_type}"
    
    train_cmd="python -m czi.ai.rbio.baselines.train_MLP \
      --train-dataset-path ${train_file} \
      --num-epochs 30 \
      --batch-size 32 \
      --embedding-file ${embedding_file} \
      --checkpoint-dir ${checkpoint_dir}"

    echo "Executing training: $train_cmd"
    eval $train_cmd
    echo
  done
done

# Then, test each trained model on all test datasets
echo -e "\nTesting phase..."
for train in "${datasets[@]}"; do
  for test in "${datasets[@]}"; do
    for i in "${!embedding_types[@]}"; do
      embedding_type="${embedding_types[$i]}"
      embedding_file="${EMBEDDING_PATHS[$i]}"
      
      test_file="${DATASET_BASE}/${test}-test-v0.2.0-no-augmentation.csv"
      output_file="${OUTPUT_BASE}/${train}-${test}-${embedding_type}.csv"
      checkpoint_dir="${CHECKPOINT_BASE}/MLP-${train}-${embedding_type}"

      # Skip if model doesn't exist
      if [ ! -f "${checkpoint_dir}/mlp_model.pt" ]; then
        echo "Model not found for train=${train}, embedding=${embedding_type}, skipping testing..."
        continue
      fi

      echo "Testing model trained on ${train} with ${embedding_type} embeddings on ${test} test set"
      
      test_cmd="python -m czi.ai.rbio.baselines.test_MLP \
        --test-dataset-path ${test_file} \
        --mlp-model-path ${checkpoint_dir}/mlp_model.pt \
        --embedding-file ${embedding_file} \
        --output-csv-path ${output_file} \
        --batch-size 32"

      echo "Executing testing: $test_cmd"
      eval $test_cmd
      echo "----------------------------------------"
    done
  done
done
