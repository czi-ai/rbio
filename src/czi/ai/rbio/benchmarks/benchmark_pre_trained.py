import os
from pathlib import Path
from typing import Dict, List, Any

import click
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from czi.ai.rbio.data.datasets import RbioDataset
from czi.ai.rbio.utils.utils import extract_answer


def benchmark_pretrained(
    dataset_path: os.PathLike,
    model_name: str,
    output_path: os.PathLike,
    batch_size: int = 8,
) -> None:

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set left padding for decoder-only architecture
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_csv(dataset_path)
    rbio_dataset = RbioDataset(dataset, tokenizer)
    dataloader = DataLoader(rbio_dataset, batch_size=batch_size, shuffle=False)

    # Initialize list to store results
    results: List[Dict[str, Any]] = []

    for batch_idx, (texts, labels) in enumerate(tqdm(dataloader)):
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for text, response, label in zip(texts, responses, labels):
            answer = extract_answer(response)
            bool_label = label == 1

            # Record result
            result = {
                "prompt": text,
                "completion": response,
                "answer": answer,
                "binary_answer": 1 if answer is True else (0 if answer is False else -1),
                "ground_truth": bool_label,
            }
            results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


@click.command()
@click.option("--dataset-path", help="Dataset CSV file path", required=True)
@click.option("--model-name", help="Huggingface model name", required=True)
@click.option("--output-path", help="Path to save results (CSV file)", required=True)
@click.option("--batch-size", help="Batch size for inference", default=8, type=int)
def benchmark(
    dataset_path: os.PathLike,
    model_name: str,
    output_path: os.PathLike,
    batch_size: int,
):
    benchmark_pretrained(
        dataset_path=dataset_path,
        model_name=model_name,
        output_path=output_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    benchmark()
