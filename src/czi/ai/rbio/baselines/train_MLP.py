import os
import pickle
import random

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from czi.ai.rbio.data.datasets import BalancedBatchSampler, GeneDataset
from czi.ai.rbio.model.models import MLPClassifier
from czi.ai.rbio.utils.utils import compute_embeddings_hash


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    train_df: pd.DataFrame,
    emb_dict: dict,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    train_dataset = GeneDataset(train_df, emb_dict)
    sampler = BalancedBatchSampler(
        train_dataset.pos_indices, train_dataset.neg_indices, batch_size
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    input_dim = len(next(iter(emb_dict.values())))
    model = MLPClassifier(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for gene_pert, gene_mon, label in train_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.to(device).unsqueeze(1)

            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * gene_pert.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def embedding_training(
    training_set_paths,
    emb_dict: dict,
    batch_size: int,
    num_epochs: int,
    checkpoint_dir: os.PathLike,
):
    dfs = [pd.read_csv(path) for path in training_set_paths]
    df_training = pd.concat(dfs, ignore_index=True)

    genes = pd.unique(df_training[["gene_perturbed", "gene_monitored"]].values.ravel())
    all_genes = sorted(set(genes))

    # Verify all genes are in emb_dict
    missing_genes = [gene for gene in all_genes if gene.lower() not in emb_dict]
    if missing_genes:
        raise ValueError(f"Missing embeddings for genes: {missing_genes}")

    model = train_model(
        df_training, emb_dict, batch_size=batch_size, num_epochs=num_epochs
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mlp_model.pt")
    embeddings_hash_path = os.path.join(checkpoint_dir, "embeddings_hash.txt")

    # Save model and embeddings hash
    torch.save(model.state_dict(), checkpoint_path)
    embeddings_hash = compute_embeddings_hash(emb_dict)
    with open(embeddings_hash_path, "w") as f:
        f.write(embeddings_hash)

    print(f"Model checkpoint saved to {checkpoint_path}")
    print(f"Embeddings hash saved to {embeddings_hash_path}")


@click.command()
@click.option(
    "--train-dataset-path",
    required=True,
    multiple=True,
    help="Training dataset CSV file(s)",
)
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--num-epochs", default=10, help="Number of training epochs")
@click.option("--embedding-file", required=True, help="Path to embedding .pkl file")
@click.option(
    "--checkpoint-dir", required=True, help="Directory to save model checkpoint"
)
def main(
    train_dataset_path,
    batch_size,
    num_epochs,
    embedding_file,
    checkpoint_dir,
):
    set_seed(42)
    with open(embedding_file, "rb") as f:
        emb_dict = pickle.load(f)
    embedding_training(
        train_dataset_path, emb_dict, batch_size, num_epochs, checkpoint_dir
    )


if __name__ == "__main__":
    main()
