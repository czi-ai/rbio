import os
import random
from typing import Tuple

import click
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Output: logits (use with BCEWithLogitsLoss)


# Custom dataset
class GeneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, name_to_embedding: dict):
        self.df = df.reset_index(drop=True)
        self.name_to_embedding = name_to_embedding

        # Indexes for positive and negative samples
        self.pos_indices = self.df[self.df["label"] == 1].index.tolist()
        self.neg_indices = self.df[self.df["label"] == 0].index.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        gene_pert = torch.tensor(
            self.name_to_embedding[row["gene_perturbed"]], dtype=torch.float32
        )
        gene_mon = torch.tensor(
            self.name_to_embedding[row["gene_monitored"]], dtype=torch.float32
        )
        label = torch.tensor(row["label"], dtype=torch.float32)
        return gene_pert, gene_mon, label


class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size):
        super().__init__()
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

    def __iter__(self) -> list:
        pos_pool = random.sample(self.pos_indices, len(self.pos_indices))
        neg_pool = random.sample(self.neg_indices, len(self.neg_indices))
        min_len = min(len(pos_pool), len(neg_pool))

        for i in range(0, min_len, self.half_batch):
            pos_batch = pos_pool[i : i + self.half_batch]
            neg_batch = neg_pool[i : i + self.half_batch]
            if len(pos_batch) == self.half_batch and len(neg_batch) == self.half_batch:
                batch = pos_batch + neg_batch
                random.shuffle(batch)
                yield batch

    def __len__(self) -> int:
        return min(len(self.pos_indices), len(self.neg_indices)) // self.half_batch


def train_model(
    train_df: pd.DataFrame,
    name_to_embedding: dict,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    # Dataset and loader
    train_dataset = GeneDataset(train_df, name_to_embedding)
    sampler = BalancedBatchSampler(
        train_dataset.pos_indices, train_dataset.neg_indices, batch_size
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    input_dim = len(
        next(iter(name_to_embedding.values()))
    )  # infer input dim from one embedding
    model = MLPClassifier(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for gene_pert, gene_mon, label in train_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.to(device).unsqueeze(1)  # match shape (batch_size, 1)

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


def one_hot_gene_perturbation(
    training_set_path: os.PathLike,
    test_set_path: os.PathLike,
    batch_size: int = 32,
    num_epochs: int = 10,
    output_csv_path: os.PathLike = "./results.csv",
) -> None:
    df_training = pd.read_csv(training_set_path)
    df_testing = pd.read_csv(test_set_path)

    # Collect all unique gene names from both gene_perturbed and gene_monitored columns
    genes_train = pd.unique(
        df_training[["gene_perturbed", "gene_monitored"]].values.ravel()
    )
    genes_test = pd.unique(
        df_testing[["gene_perturbed", "gene_monitored"]].values.ravel()
    )

    all_genes = sorted(
        set(genes_train).union(set(genes_test))
    )  # ensure deterministic ordering
    print(f"Found {len(all_genes)} unique gene names across training and testing sets.")

    # Build 1-hot encoding dictionary
    gene_to_index = {gene: i for i, gene in enumerate(all_genes)}
    identity_matrix = np.eye(len(all_genes), dtype=np.float32)
    name_to_embedding = {
        gene: identity_matrix[idx] for gene, idx in gene_to_index.items()
    }

    # Train the model
    model = train_model(
        df_training, name_to_embedding, batch_size=batch_size, num_epochs=num_epochs
    )

    test_model(
        model,
        df_testing,
        name_to_embedding,
        output_csv=output_csv_path,
        batch_size=batch_size,
    )

    print(f"Saved results to {output_csv_path}")


def embedding_gene_perturbation(
    training_set_path: os.PathLike,
    test_set_path: os.PathLike,
    emb_dict: dict,
    batch_size: int = 32,
    num_epochs: int = 10,
    output_csv_path: os.PathLike = "./results.csv",
):
    df_training = pd.read_csv(training_set_path)
    df_testing = pd.read_csv(test_set_path)

    # Collect all unique gene names from both gene_perturbed and gene_monitored columns
    genes_train = pd.unique(
        df_training[["gene_perturbed", "gene_monitored"]].values.ravel()
    )
    genes_test = pd.unique(
        df_testing[["gene_perturbed", "gene_monitored"]].values.ravel()
    )

    all_genes = sorted(
        set(genes_train).union(set(genes_test))
    )  # ensure deterministic ordering
    print(f"Found {len(all_genes)} unique gene names across training and testing sets.")

    # Build 1-hot encoding dictionary
    gene_to_index = {gene: i for i, gene in enumerate(all_genes)}

    name_to_embedding = {}
    missing = 0

    for gene, idx in gene_to_index.items():
        try:
            name_to_embedding[gene] = np.asarray(
                emb_dict[gene.lower()], dtype=np.float32
            )
        except KeyError:
            missing += 1
            print(
                f"WARNING: the embedding for gene {gene} is not in the dict, total missing {missing}"
            )
            first_emb_dict = np.asarray(emb_dict[list(emb_dict.keys())[0]])
            rand_embedding = np.random.randn(first_emb_dict.shape[0]).astype(np.float32)
            name_to_embedding[gene] = rand_embedding

    # Train the model
    model = train_model(
        df_training, name_to_embedding, batch_size=batch_size, num_epochs=num_epochs
    )

    test_model(
        model,
        df_testing,
        name_to_embedding,
        output_csv=output_csv_path,
        batch_size=batch_size,
    )

    print(f"Saved results to {output_csv_path}")


@click.command()
@click.option(
    "--train-dataset-path", help="Dataset CSV file path", required=True, multiple=False
)
@click.option(
    "--test-dataset-path", help="Dataset CSV file path", required=True, multiple=False
)
@click.option(
    "--strategy",
    help="Whether we should use 1-hot-encoded gene representation or gene embeddings",
    required=True,
)
@click.option("--batch-size", help="Batch-size", default=32)
@click.option("--num-epochs", help="Number of epochs", default=10)
@click.option("--embedding-file", help="Embedding file", default=None)
@click.option("--output-csv-path", help="Output CSV file path", required=True)
def train(
    train_dataset_path: os.PathLike,
    test_dataset_path: os.PathLike,
    strategy: str,
    batch_size: int,
    num_epochs: int,
    embedding_file: os.PathLike,
    output_csv_path: os.PathLike,
):
    set_seed(42)

    if strategy == "1-hot":
        one_hot_gene_perturbation(
            train_dataset_path,
            test_dataset_path,
            batch_size,
            num_epochs,
            output_csv_path,
        )
    else:
        with open(embedding_file, "rb") as f:
            emb_dict = pickle.load(f)

        embedding_gene_perturbation(
            train_dataset_path,
            test_dataset_path,
            emb_dict,
            batch_size,
            num_epochs,
            output_csv_path,
        )


if __name__ == "__main__":
    train()
