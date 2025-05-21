import os
import pickle
from typing import Tuple

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from czi.ai.rbio.data.datasets import GeneDataset
from czi.ai.rbio.model.models import MLPClassifier
from czi.ai.rbio.utils.utils import compute_embeddings_hash


def test_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    name_to_embedding: dict,
    output_csv: os.PathLike,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    test_dataset = GeneDataset(test_df, name_to_embedding)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    results = []
    idx = 0  # Track index in test_df to match gene names

    with torch.no_grad():
        for gene_pert, gene_mon, label in test_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.cpu().numpy().flatten()

            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            for gt, pred in zip(label, preds):
                results.append(
                    {
                        "prompt": "",
                        "completion": "",
                        "answer": pred,
                        "binary_answer": int(pred),
                        "ground_truth": int(gt),
                        "gene_perturbed": test_df.iloc[idx]["gene_perturbed"],
                        "gene_monitored": test_df.iloc[idx]["gene_monitored"],
                    }
                )
                idx += 1

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nPrediction CSV saved to: {output_csv}")


@click.command()
@click.option(
    "--test-dataset-path",
    help="Test dataset CSV file path",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--mlp-model-path",
    help="Path to the trained MLP model checkpoint",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--embedding-file",
    help="Path to the gene embedding dictionary pickle file",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--output-csv-path",
    help="Output CSV file path for predictions",
    required=True,
)
@click.option(
    "--batch-size",
    help="Batch size for testing",
    default=32,
)
def main(
    test_dataset_path: os.PathLike,
    mlp_model_path: os.PathLike,
    embedding_file: os.PathLike,
    output_csv_path: os.PathLike,
    batch_size: int,
):
    # Load test dataset
    test_df = pd.read_csv(test_dataset_path)

    # Load model and embeddings
    with open(embedding_file, "rb") as f:
        name_to_embedding = pickle.load(f)

    # Check embeddings hash
    embeddings_hash_path = os.path.join(os.path.dirname(mlp_model_path), "embeddings_hash.txt")
    if os.path.exists(embeddings_hash_path):
        with open(embeddings_hash_path, "r") as f:
            expected_hash = f.read().strip()
        current_hash = compute_embeddings_hash(name_to_embedding)
        if current_hash != expected_hash:
            print("\033[93mWARNING: Embeddings hash does not match! Results will be random.\033[0m")
            print(f"Expected hash: {expected_hash}")
            print(f"Current hash:  {current_hash}")

    input_dim = len(next(iter(name_to_embedding.values())))
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    model.eval()

    # Run testing
    test_model(
        model,
        test_df,
        name_to_embedding,
        output_csv=output_csv_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
