import pandas as pd
import torch
from torch.utils.data import Dataset
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return text, label
