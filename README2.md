# RBIO CLI Guide

This guide explains how to use the various command-line interfaces (CLIs) provided by the RBIO package for training, benchmarking, and evaluating models.

## Installation

First, install the package in development mode:

```bash
# For development
pip install -e .

# For production
pip install .
```

## Training

The training CLI allows you to train models using the GRPO algorithm. Here's how to use it:

```bash
python -m czi.ai.rbio.train \
    --dataset-path <path_to_dataset> \
    --model-name <huggingface_model_name> \
    --checkpoint-dir <output_directory> \
    --resume <true/false> \
    --batch-size <batch_size> \
    --n-generations <num_generations>
```

### Example

```bash
python -m czi.ai.rbio.train \
    --dataset-path /mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/hepg2-train-v0.1.1-no-augmentation.csv \
    --dataset-path /mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/jurkat-train-v0.1.1-no-augmentation.csv \
    --dataset-path /mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-train-v0.1.1-no-augmentation.csv \
    --dataset-path /mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/rpe1-train-v0.1.1-no-augmentation.csv \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --checkpoint-dir /mnt/czi-sci-ai/project-rbio/checkpoints/PertQA-DE/K562/1_Rewrite/Qwen25-3b-Instruct-NORA-AllData \
    --resume false \
    --batch-size 4 \
    --n-generations 4
```

### Parameters

- `--dataset-path`: Path to the training dataset CSV file (can be specified multiple times)
- `--model-name`: Name of the Hugging Face model to use
- `--checkpoint-dir`: Directory to save model checkpoints
- `--resume`: Whether to resume training from a checkpoint (true/false)
- `--batch-size`: Training batch size (default: 4)
- `--n-generations`: Number of generations for GRPO (default: 4)

## Benchmarking

### Benchmark Pre-trained Models

```bash
python -m czi.ai.rbio.benchmarks.benchmark_pre_trained \
    --dataset-path <path_to_dataset> \
    --model-name <huggingface_model_name> \
    --output-path <output_csv_path> \
    --batch-size <batch_size>
```

### Benchmark GRPO-trained Models

```bash
python -m czi.ai.rbio.benchmarks.benchmark_grpo_trained \
    --dataset-path <path_to_dataset> \
    --model-name <huggingface_model_name> \
    --grpo-checkpoint <path_to_checkpoint> \
    --output-path <output_csv_path> \
    --batch-size <batch_size>
```

### Benchmark Commercial LLMs

```bash
python -m czi.ai.rbio.benchmarks.benchmark_commercial_llm \
    --dataset-path <path_to_dataset> \
    --llm-model <model_name> \
    --llm-endpoint <endpoint_url> \
    --llm-api-key <api_key> \
    --output-path <output_csv_path>
```

### Parameters

- `--dataset-path`: Path to the benchmark dataset CSV file
- `--model-name`: Name of the Hugging Face model to use
- `--grpo-checkpoint`: Path to the trained model checkpoint (for GRPO-trained models)
- `--output-path`: Path to save benchmark results
- `--batch-size`: Inference batch size (default: 8)
- `--llm-model`: Name of the commercial LLM model
- `--llm-endpoint`: URL of the commercial model endpoint
- `--llm-api-key`: API key for the commercial model endpoint

## Processing Benchmark Results

After running benchmarks, you can analyze the results using:

```bash
python -m czi.ai.rbio.utils.process_benchmark \
    --results-csv <path_to_benchmark_results>
```

This will output:
- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)
- Accuracy
- Precision
- Recall
- F1 Score
- AUC ROC

### Example

```bash
python -m czi.ai.rbio.utils.process_benchmark \
    --results-csv /path/to/benchmark_results.csv
```

## Notes

1. All paths can be absolute or relative
2. For commercial LLM benchmarking, ensure you have the necessary environment variables set:
   - `LLM_ENDPOINT_URL`
   - `LLM_ENDPOINT_KEY`
3. The benchmark results CSV will contain:
   - `prompt`: The input prompt
   - `completion`: The model's response
   - `answer`: The extracted answer
   - `binary_answer`: The binary representation of the answer (1 for True, 0 for False, -1 for None)
   - `ground_truth`: The true label 