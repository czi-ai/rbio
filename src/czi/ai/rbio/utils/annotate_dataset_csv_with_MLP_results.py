import os
import pickle
from typing import List

import click
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from czi.ai.rbio.data.datasets import GeneDataset
from czi.ai.rbio.model.models import MLPClassifier
from czi.ai.rbio.utils.utils import compute_embeddings_hash


def annotate_dataset_with_mlp(
    dataset_path: os.PathLike,
    mlp_model_path: os.PathLike,
    embedding_file: os.PathLike,
    output_path: os.PathLike,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Annotates a dataset CSV with MLP model predictions by updating the 'confidence' column
    while keeping the 'label' column as "no|yes".
    
    Args:
        dataset_path: Path to the input dataset CSV
        mlp_model_path: Path to the trained MLP model checkpoint
        embedding_file: Path to the gene embedding dictionary pickle file
        output_path: Path where to save the annotated dataset
        batch_size: Batch size for inference
        device: Device to run inference on
    """
    # Load dataset
    dataset_df = pd.read_csv(dataset_path)
    
    # Load embeddings
    with open(embedding_file, "rb") as f:
        emb_dict = pickle.load(f)
    
    # Verify all genes are in emb_dict
    genes = pd.unique(dataset_df[["gene_perturbed", "gene_monitored"]].values.ravel())
    missing_genes = [gene for gene in genes if gene.lower() not in emb_dict]
    if missing_genes:
        raise ValueError(f"Missing embeddings for genes: {missing_genes}")
    
    # Check embeddings hash
    embeddings_hash_path = os.path.join(os.path.dirname(mlp_model_path), "embeddings_hash.txt")
    if os.path.exists(embeddings_hash_path):
        with open(embeddings_hash_path, "r") as f:
            expected_hash = f.read().strip()
        current_hash = compute_embeddings_hash(emb_dict)
        if current_hash != expected_hash:
            print("\033[93mWARNING: Embeddings hash does not match! Results will be random.\033[0m")
            print(f"Expected hash: {expected_hash}")
            print(f"Current hash:  {current_hash}")
    
    # Load model
    input_dim = len(next(iter(emb_dict.values())))
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = GeneDataset(dataset_df, emb_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run inference
    probabilities: List[float] = []
    with torch.no_grad():
        for gene_pert, gene_mon, _ in dataloader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            
            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().numpy().flatten())
    
    # Update confidence column with probabilities
    # Format: "1-prob|prob" for each row
    dataset_df["class_confidences"] = [f"{1-prob:.4f}|{prob:.4f}" for prob in probabilities]
    dataset_df["label"] = [int(prob > 0.5) for prob in probabilities]
    
    # Ensure classes column is "no|yes" for all rows
    dataset_df["classes"] = "no|yes"
    
    # Save annotated dataset
    dataset_df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to: {output_path}")


@click.command()
@click.option(
    "--dataset-path",
    required=True,
    help="Path to the input dataset CSV file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--mlp-model-path",
    required=True,
    help="Path to the trained MLP model checkpoint",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--embedding-file",
    required=True,
    help="Path to the gene embedding dictionary pickle file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output-path",
    required=True,
    help="Path where to save the annotated dataset",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size for inference",
)
def main(
    dataset_path: os.PathLike,
    mlp_model_path: os.PathLike,
    embedding_file: os.PathLike,
    output_path: os.PathLike,
    batch_size: int,
):
    annotate_dataset_with_mlp(
        dataset_path=dataset_path,
        mlp_model_path=mlp_model_path,
        embedding_file=embedding_file,
        output_path=output_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main() 
