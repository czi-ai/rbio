from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint
import pandas as pd
from tqdm import tqdm
import os
import re
import click
import torch
from torch.utils.data import DataLoader
from czi.ai.rbio.data.datasets import RbioDataset


def extract_answer(text):
    found = re.search(r"<answer>\s*(yes|no)\s*</answer>", text, re.IGNORECASE)
    if found:
        if found.group(1).strip().lower() == "yes":
            return True
        if found.group(1).strip().lower() == "no":
            return False

    return None


def benchmark_grpo_trained(
    dataset_path: os.PathLike,
    model_name: str,
    model_checkpoint: os.PathLike,
    batch_size: int = 8,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_sharded_checkpoint(model, model_checkpoint, strict=False)

    dataset = pd.read_csv(dataset_path)
    grpo_dataset = RbioDataset(dataset, tokenizer)
    dataloader = DataLoader(grpo_dataset, batch_size=batch_size, shuffle=False)

    stats = {
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "tn": 0,
        "unanswered": 0,
    }

    for batch_idx, (texts, labels) in enumerate(tqdm(dataloader)):
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for response, label in zip(responses, labels):
            answer = extract_answer(response)
            bool_label = label == 1

            if answer is not None:
                if answer == True and bool_label == True:
                    stats["tp"] += 1
                elif answer == True and bool_label == False:
                    stats["fp"] += 1
                elif answer == False and bool_label == False:
                    stats["tn"] += 1
                elif answer == False and bool_label == True:
                    stats["fn"] += 1
            else:
                stats["unanswered"] += 1

        if batch_idx % 10 == 0:  # Print stats every 10 batches
            print(f"Partial results @ batch {batch_idx}: {stats}")

    print(f"STATS HAVE BEEN GENERATED FOR DATASET {dataset_path}")
    print(stats)
    print(f"DONE WITH {model_name}::{model_checkpoint}")

    return stats


@click.command()
@click.option("--dataset-path", help="Dataset CSV file path", required=True)
@click.option("--model-name", help="Huggingface model name", required=True)
@click.option("--grpo-checkpoint", help="Path of trained model checkpoint", required=True)
@click.option("--batch-size", help="Batch size for inference", default=8, type=int)
def benchmark(dataset_path: os.PathLike, model_name: str, grpo_checkpoint: os.PathLike, batch_size: int):
    benchmark_grpo_trained(
        dataset_path=dataset_path,
        model_name=model_name,
        model_checkpoint=grpo_checkpoint,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    benchmark()
