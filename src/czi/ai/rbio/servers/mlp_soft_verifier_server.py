import os
import pickle

import click
import torch
from flask import Flask, jsonify, request
from torch import nn

app = Flask(__name__)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Global variables to store model and embeddings
model = None
name_to_embedding = None


@app.route("/perturbation", methods=["POST"])
def perturbation():
    data = request.get_json()
    gene_a = data.get("Gene_A")
    gene_b = data.get("Gene_B")

    if gene_a not in name_to_embedding or gene_b not in name_to_embedding:
        return (
            jsonify(
                {
                    "error": f"Gene not found: {gene_a if gene_a not in name_to_embedding else gene_b}"
                }
            ),
            400,
        )

    emb_a = torch.tensor(
        name_to_embedding[gene_a.lower()], dtype=torch.float32
    ).unsqueeze(0)
    emb_b = torch.tensor(
        name_to_embedding[gene_b.lower()], dtype=torch.float32
    ).unsqueeze(0)

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
    "--gene-dict-path",
    required=True,
    help="Path to the gene embedding dictionary pickle file",
    type=click.Path(exists=True, dir_okay=False),
)
def main(mlp_model_path: str, gene_dict_path: str):
    global model, name_to_embedding

    # Load embedding dictionary
    with open(gene_dict_path, "rb") as f:
        name_to_embedding = pickle.load(f)

    # Infer input dimension and load model
    input_dim = len(next(iter(name_to_embedding.values())))
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    model.eval()

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
