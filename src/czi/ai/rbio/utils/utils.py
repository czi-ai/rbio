import hashlib
import json
import re


def extract_answer(text):
    found = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    if found:
        return found.group(1).strip().lower()

    return ""


def extract_think(text, separator="\n"):
    think_contents = re.findall(
        r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
    )
    return separator.join(think_contents).strip()


def compute_embeddings_hash(emb_dict: dict) -> str:
    # Convert embeddings to a stable string representation
    emb_str = json.dumps({k: v.tolist() for k, v in sorted(emb_dict.items())})
    return hashlib.md5(emb_str.encode()).hexdigest()
