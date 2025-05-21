import os
import pickle
import random

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GeneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, name_to_embedding: dict):
        self.df = df.reset_index(drop=True)
        self.name_to_embedding = name_to_embedding
        self.pos_indices = self.df[self.df["label"] == 1].index.tolist()
        self.neg_indices = self.df[self.df["label"] == 0].index.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gene_pert = torch.tensor(
            self.name_to_embedding[row["gene_perturbed"].lower()], dtype=torch.float32
        )
        gene_mon = torch.tensor(
            self.name_to_embedding[row["gene_monitored"].lower()], dtype=torch.float32
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

    def __iter__(self):
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

    def __len__(self):
        return min(len(self.pos_indices), len(self.neg_indices)) // self.half_batch


def train_model(
    train_df: pd.DataFrame,
    name_to_embedding: dict,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    train_dataset = GeneDataset(train_df, name_to_embedding)
    sampler = BalancedBatchSampler(
        train_dataset.pos_indices, train_dataset.neg_indices, batch_size
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    input_dim = len(next(iter(name_to_embedding.values())))
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
    gene_to_index = {gene: i for i, gene in enumerate(all_genes)}

    name_to_embedding = {}
    missing = 0
    for gene, idx in gene_to_index.items():
        try:
            name_to_embedding[gene.lower()] = np.asarray(
                emb_dict[gene.lower()], dtype=np.float32
            )
        except KeyError:
            missing += 1
            print(f"WARNING: Missing embedding for gene {gene} (#{missing})")
            rand_emb = np.random.randn(len(next(iter(emb_dict.values())))).astype(
                np.float32
            )
            name_to_embedding[gene.lower()] = rand_emb

    model = train_model(
        df_training, name_to_embedding, batch_size=batch_size, num_epochs=num_epochs
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mlp_model.pt")
    name_to_embedding_path = os.path.join(checkpoint_dir, "name_to_embedding.pkl")

    torch.save(model.state_dict(), checkpoint_path)
    with open(name_to_embedding_path, "wb") as f:
        pickle.dump(name_to_embedding, f)

    print(f"Model checkpoint saved to {checkpoint_path}")
    print(f"Embedding dictionary saved to {name_to_embedding_path}")


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
