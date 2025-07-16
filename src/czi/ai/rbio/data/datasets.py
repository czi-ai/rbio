import random

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer


class RbioDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        row = self.dataframe.iloc[idx]
        system_prompt = row["system_prompt"]
        user_prompt = row["user_prompt"]
        label = row["label"]
        gene_perturbed = row["gene_perturbed"]
        gene_monitored = row["gene_monitored"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return text, label, gene_perturbed, gene_monitored


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
