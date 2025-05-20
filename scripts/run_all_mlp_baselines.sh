#!/bin/bash

# Define datasets and strategies
datasets=(k562 hepg2 jurkat rpe1)
strategies=("1-hot" "gene2vec" "esm_emb")

# Paths
DATASET_BASE="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE"
OUTPUT_BASE="/mnt/czi-sci-ai/project-rbio/baselines"
CHECKPOINT_BASE="/mnt/czi-sci-ai/project-rbio/checkpoints/MLP"
GENE2VEC_EMBEDDING="/mnt/czi-sci-ai/project-rbio/repr/gene2vec_embeddings_filled.pkl"
ESM_EMBEDDING="/mnt/czi-sci-ai/project-rbio/repr/esm_embedding_dictionary_filled.pkl"

# First, train models for each training dataset and strategy
echo "Training phase..."
for train in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    train_file="${DATASET_BASE}/${train}-train-v0.2.0-no-augmentation.csv"
    checkpoint_dir="${CHECKPOINT_BASE}/MLP-${train}-${strategy}"

    # Skip if model already exists
    if [ -f "${checkpoint_dir}/mlp_model.pt" ]; then
      echo "Model already exists for train=${train}, strategy=${strategy}, skipping training..."
      continue
    fi

    echo "Training model for: train=${train}, strategy=${strategy}"
    
    train_cmd="python -m czi.ai.rbio.baselines.train_MLP \
      --train-dataset-path ${train_file} \
      --strategy ${strategy} \
      --num-epochs 30 \
      --batch-size 32 \
      --checkpoint-dir ${checkpoint_dir}"

    # Add appropriate embedding file based on strategy
    if [ "$strategy" == "gene2vec" ]; then
      train_cmd="${train_cmd} --embedding-file ${GENE2VEC_EMBEDDING}"
    elif [ "$strategy" == "esm_emb" ]; then
      train_cmd="${train_cmd} --embedding-file ${ESM_EMBEDDING}"
    fi

    echo "Executing training: $train_cmd"
    eval $train_cmd
    echo
  done
done

# Then, test each trained model on all test datasets
echo -e "\nTesting phase..."
for train in "${datasets[@]}"; do
  for test in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
      test_file="${DATASET_BASE}/${test}-test-v0.2.0-no-augmentation.csv"
      output_file="${OUTPUT_BASE}/${train}-${test}-${strategy}.csv"
      checkpoint_dir="${CHECKPOINT_BASE}/MLP-${train}-${strategy}"

      # Skip if model doesn't exist
      if [ ! -f "${checkpoint_dir}/mlp_model.pt" ]; then
        echo "Model not found for train=${train}, strategy=${strategy}, skipping testing..."
        continue
      fi

      echo "Testing model trained on ${train} with ${strategy} strategy on ${test} test set"
      
      test_cmd="python -m czi.ai.rbio.baselines.test_MLP \
        --test-dataset-path ${test_file} \
        --mlp-model-path ${checkpoint_dir}/mlp_model.pt \
        --gene-dict-path ${checkpoint_dir}/name_to_embedding.pkl \
        --output-csv-path ${output_file} \
        --batch-size 32"

      echo "Executing testing: $test_cmd"
      eval $test_cmd
      echo "----------------------------------------"
    done
  done
done
