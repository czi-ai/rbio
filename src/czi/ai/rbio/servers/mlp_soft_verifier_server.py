import os
import pickle

import click
import torch
from flask import Flask, jsonify, request
from torch import nn

from czi.ai.rbio.model.models import MLPClassifier
from czi.ai.rbio.utils.utils import compute_embeddings_hash

app = Flask(__name__)

# Global variables to store model and embeddings
model = None
emb_dict = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.route("/perturbation", methods=["POST"])
def perturbation():
    data = request.get_json()
    gene_a = data.get("Gene_A")
    gene_b = data.get("Gene_B")

    if gene_a.lower() not in emb_dict or gene_b.lower() not in emb_dict:
        return (
            jsonify(
                {
                    "error": f"Gene not found: {gene_a if gene_a.lower() not in emb_dict else gene_b}"
                }
            ),
            400,
        )

    emb_a = (
        torch.tensor(emb_dict[gene_a.lower()], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    emb_b = (
        torch.tensor(emb_dict[gene_b.lower()], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    inputs = torch.cat([emb_a, emb_b], dim=1)
    with torch.no_grad():
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()

    return jsonify(
        {"Gene_A": gene_a, "Gene_B": gene_b, "perturbation_probability": prob}
    )


@click.command()
@click.option(
    "--mlp-model-path",
    required=True,
    help="Path to the MLP model checkpoint file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--embedding-file",
    required=True,
    help="Path to the gene embedding dictionary pickle file",
    type=click.Path(exists=True, dir_okay=False),
)
def main(mlp_model_path: str, embedding_file: str):
    global model, emb_dict

    # Load embedding dictionary
    with open(embedding_file, "rb") as f:
        emb_dict = pickle.load(f)

    # Check embeddings hash
    embeddings_hash_path = os.path.join(
        os.path.dirname(mlp_model_path), "embeddings_hash.txt"
    )
    if os.path.exists(embeddings_hash_path):
        with open(embeddings_hash_path, "r") as f:
            expected_hash = f.read().strip()
        current_hash = compute_embeddings_hash(emb_dict)
        if current_hash != expected_hash:
            print(
                "\033[93mWARNING: Embeddings hash does not match! Results will be random.\033[0m"
            )
            print(f"Expected hash: {expected_hash}")
            print(f"Current hash:  {current_hash}")

    # Infer input dimension and load model
    input_dim = len(next(iter(emb_dict.values())))
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    model = model.to(device)  # Move model to GPU if available
    model.eval()

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
