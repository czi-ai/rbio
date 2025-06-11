import os
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint

from czi.ai.rbio.data.datasets import RbioDataset
from czi.ai.rbio.utils.utils import extract_answer


def benchmark_grpo_trained(
    dataset_path: os.PathLike,
    model_name: str,
    model_checkpoint: os.PathLike,
    output_path: os.PathLike,
    batch_size: int = 8,
) -> None:

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_sharded_checkpoint(model, model_checkpoint, strict=False)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_csv(dataset_path)
    # .sample(5000)
    grpo_dataset = RbioDataset(dataset, tokenizer)
    dataloader = DataLoader(grpo_dataset, batch_size=batch_size, shuffle=False)

    # Initialize list to store results
    results: List[Dict[str, Any]] = []

    for batch_idx, (texts, labels, genes_perturbed, genes_monitored) in enumerate(
        tqdm(dataloader)
    ):
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for text, response, label, gene_perturbed, gene_monitored in zip(
            texts, responses, labels, genes_perturbed, genes_monitored
        ):
            answer = extract_answer(response)
            bool_label = label == 1

            # Record result
            result = {
                "prompt": text,
                "completion": response,
                "answer": answer,
                "binary_answer": (
                    1 if answer is True else (0 if answer is False else -1)
                ),
                "ground_truth": bool_label.item(),
                "gene_perturbed": gene_perturbed,
                "gene_monitored": gene_monitored,
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
@click.option(
    "--grpo-checkpoint", help="Path of trained model checkpoint", required=True
)
@click.option("--output-path", help="Path to save results (CSV file)", required=True)
@click.option("--batch-size", help="Batch size for inference", default=8, type=int)
def benchmark(
    dataset_path: os.PathLike,
    model_name: str,
    grpo_checkpoint: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
):
    benchmark_grpo_trained(
        dataset_path=dataset_path,
        model_name=model_name,
        model_checkpoint=grpo_checkpoint,
        output_path=output_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    benchmark()
