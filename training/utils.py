import hashlib
import json
import os
import pickle
import random

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig


def compute_embeddings_hash(emb_dict: dict) -> str:
    """Compute hash of embeddings dictionary for consistency checking"""
    # Convert embeddings to a stable string representation
    emb_str = json.dumps({k: v.tolist() for k, v in sorted(emb_dict.items())})
    return hashlib.md5(emb_str.encode()).hexdigest()


def load_mlp_classifier(mlp_model_path: str, embedding_file: str, obj_type):
    """Load MLP model and embeddings from disk with hash verification"""
    global mlp_model, embeddings_dict

    print(f"Loading MLP model from: {mlp_model_path}")
    print(f"Loading embeddings from: {embedding_file}")

    # Load embeddings
    with open(embedding_file, "rb") as f:
        embeddings_dict = pickle.load(f)

    # Check embeddings hash if available
    embeddings_hash_path = os.path.join(os.path.dirname(mlp_model_path), "embeddings_hash.txt")
    if os.path.exists(embeddings_hash_path):
        with open(embeddings_hash_path, "r") as f:
            expected_hash = f.read().strip()
        current_hash = compute_embeddings_hash(embeddings_dict)

        if current_hash != expected_hash:
            print("\033[93mWARNING: Embeddings hash does not match! Results may be unreliable.\033[0m")
            print(f"Expected hash: {expected_hash}")
            print(f"Current hash:  {current_hash}")
        else:
            print("Embeddings hash verified successfully")
    else:
        print("No embeddings hash file found - skipping verification")

    # Load MLP model
    input_dim = len(next(iter(embeddings_dict.values())))
    mlp_model = obj_type(input_dim)
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    mlp_model.eval()

    print(f"MLP model loaded successfully with input dimension: {input_dim}")

def mlp_classifier_inference(sample_data: dict) -> float:
    """Run MLP inference on a sample to get probability"""
    global mlp_model, embeddings_dict

    if mlp_model is None or embeddings_dict is None:
        raise RuntimeError("MLP model not loaded. Call load_mlp_classifier() first.")

    # Extract gene names from keywords (assuming format: "gene_A|gene_B")
    keywords = sample_data.get("keywords", "")
    if not keywords or "|" not in keywords:
        return 0.5  # Default probability if no valid genes

    gene_names = keywords.split("|")
    if len(gene_names) < 2:
        return 0.5  # Need at least 2 genes

    gene_perturbed, gene_monitored = gene_names[0], gene_names[1]

    # Get embeddings for the genes
    try:
        gene_pert_emb = embeddings_dict.get(gene_perturbed.lower(), None)
        gene_mon_emb = embeddings_dict.get(gene_monitored.lower(), None)

        if gene_pert_emb is None or gene_mon_emb is None:
            print(f"Warning: Missing embeddings for genes: {gene_perturbed}, {gene_monitored}")
            return 0.5

        # Convert to tensors and run inference
        gene_pert_tensor = torch.tensor(gene_pert_emb, dtype=torch.float32).unsqueeze(0)
        gene_mon_tensor = torch.tensor(gene_mon_emb, dtype=torch.float32).unsqueeze(0)

        # Concatenate embeddings and run model
        inputs = torch.cat([gene_pert_tensor, gene_mon_tensor], dim=1)

        with torch.no_grad():
            logits = mlp_model(inputs)
            probability = torch.sigmoid(logits).item()

        return probability

    except Exception as e:
        print(f"Error during MLP inference: {e}")
        return 0.5


def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seeds set to {seed} for reproducibility")


def setup_model_and_tokenizer(model_name: str):
    """Load and prepare the model and tokenizer for DeepSpeed ZeRO training"""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model without device_map (DeepSpeed will handle placement)
    # Use auto dtype but let accelerate config control precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"  # Let accelerate config handle precision
    )

    print(f"Model loaded successfully")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    return model, tokenizer


def create_training_config(output_dir: str, batch_size: int, num_generations: int,
                          max_steps: int, save_every: int) -> GRPOConfig:
    """Create GRPO training configuration compatible with DeepSpeed ZeRO stage 3"""
    config = GRPOConfig(
        output_dir=output_dir,
        logging_steps=250,
        logging_first_step=True,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        run_name=f"rbio_training_{max_steps}_steps",
        save_steps=save_every,
        max_steps=max_steps,
        learning_rate=1e-5,
        warmup_steps=100,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0
    )

    return config