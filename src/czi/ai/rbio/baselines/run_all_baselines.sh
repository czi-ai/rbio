#!/bin/bash

# Define datasets and strategies
datasets=(k562 hepg2 jurkat rpe1)
strategies=("1-hot" "gene2vec")

# Paths
DATASET_BASE="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE"
OUTPUT_BASE="/mnt/czi-sci-ai/project-rbio/baselines"
EMBEDDING_FILE="/opt/jupyter-envs/rbio/rbio-fmilletari-2/work/gene2vec_embeddings.pkl"  # adjust if needed

# Loop over all combinations
for train in "${datasets[@]}"; do
  for test in "${datasets[@]}"; do

    for strategy in "${strategies[@]}"; do
      train_file="${DATASET_BASE}/${train}-train-v0.1.1-no-augmentation.csv"
      test_file="${DATASET_BASE}/${test}-test-v0.1.1-no-augmentation.csv"
      output_file="${OUTPUT_BASE}/${train}-${test}-${strategy}.csv"

      echo "Running: train=${train}, test=${test}, strategy=${strategy}"
      cmd="python -m czi.ai.rbio.baselines.simple_perturbation_classifiers \
        --train-dataset-path ${train_file} \
        --test-dataset-path ${test_file} \
        --strategy ${strategy} \
        --num-epochs 30 \
        --batch-size 32 \
        --output-csv-path ${output_file}"

      # Add embedding argument only for gene2vec
      if [ "$strategy" == "gene2vec" ]; then
        cmd="${cmd} --embedding-file ${EMBEDDING_FILE}"
      fi

      echo "Executing: $cmd"
      eval $cmd
      echo
    done
  done
done