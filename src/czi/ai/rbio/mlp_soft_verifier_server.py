import os
import pickle
import torch
from flask import Flask, request, jsonify
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


# Load model and embedding dictionary from environment variable
MLP_VERIFIER_CHECKPOINT_DIR = os.environ.get(
    "MLP_VERIFIER_CHECKPOINT_DIR", "./checkpoints"
)
MODEL_PATH = os.path.join(MLP_VERIFIER_CHECKPOINT_DIR, "mlp_model.pt")
DICT_PATH = os.path.join(MLP_VERIFIER_CHECKPOINT_DIR, "name_to_embedding.pkl")

# Load embedding dictionary
with open(DICT_PATH, "rb") as f:
    name_to_embedding = pickle.load(f)

# Infer input dimension
input_dim = len(next(iter(name_to_embedding.values())))
model = MLPClassifier(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


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

    emb_a = torch.tensor(name_to_embedding[gene_a], dtype=torch.float32).unsqueeze(0)
    emb_b = torch.tensor(name_to_embedding[gene_b], dtype=torch.float32).unsqueeze(0)

    inputs = torch.cat([emb_a, emb_b], dim=1)
    with torch.no_grad():
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()

    return jsonify(
        {"Gene_A": gene_a, "Gene_B": gene_b, "perturbation_probability": prob}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
