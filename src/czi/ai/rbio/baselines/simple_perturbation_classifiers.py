import os
import random
from typing import Union, List, Tuple

import click
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
from torch.utils.data import DataLoader, Dataset


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
        self.df = df
        self.name_to_embedding = name_to_embedding

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def test_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    name_to_embedding: dict,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    test_dataset = GeneDataset(test_df, name_to_embedding)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for gene_pert, gene_mon, label in test_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.to(device).unsqueeze(1)

            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend((probs > 0.5).int().cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC ROC": auc,
    }

    for key, val in metrics.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    return metrics


def one_hot_gene_perturbation(
    training_set_path: os.PathLike,
    test_set_path: os.PathLike,
    batch_size: int = 32,
    num_epochs: int = 10,
) -> dict:
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

    metrics = test_model(model, df_testing, name_to_embedding, batch_size=batch_size)

    return metrics


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
@click.option("--batch-size", help="Batch-size", default=4)
@click.option("--num-epochs", help="Number of epochs", default=10)
def train(
    train_dataset_path: os.PathLike,
    test_dataset_path: os.PathLike,
    strategy: str,
    batch_size: int,
    num_epochs: int,
):
    set_seed(42)

    if strategy == "1-hot":
        one_hot_gene_perturbation(
            train_dataset_path, test_dataset_path, batch_size, num_epochs
        )
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")


if __name__ == "__main__":
    train()
