import json
import logging
import os
import pickle

import click
import numpy as np
import pandas as pd
import torch
import yaml

# --- Transcriptformer imports (must be installed in your environment) ---
from hydra.utils import instantiate
from omegaconf import OmegaConf

# --- Environment variable defaults ---
TF_CFG = os.getenv(
    "TF_CFG",
    "/mnt/czi-sci-ai/project-rbio/transcriptformer/inference_config.yaml",
)
TF_MODEL_CKPT = os.getenv(
    "TF_MODEL_CKPT",
    "/mnt/czi-sci-ai/project-rbio/transcriptformer/tf_sapiens",
)
GENE2ENSEMBL_ID_FILEPATH = os.getenv(
    "GENE2ENSEMBL_ID_FILEPATH",
    "/mnt/czi-sci-ai/project-rbio/transcriptformer/gene2ensembl_ids.pkl",
)


def load_transcriptformer():
    from transcriptformer.model.embedding_surgery import change_embedding_layer
    from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings

    gene2ensembl_id = pickle.load(open(GENE2ENSEMBL_ID_FILEPATH, "rb"))

    cfg = yaml.load(open(TF_CFG, "r"), Loader=yaml.SafeLoader)
    config_path = os.path.join(cfg["model"]["checkpoint_path"], "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    mlflow_cfg = OmegaConf.create(config_dict)
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.merge(mlflow_cfg, cfg)

    cfg.model.inference_config.load_checkpoint = os.path.join(
        cfg.model.checkpoint_path, "model_weights.pt"
    )
    cfg.model.data_config.aux_vocab_path = os.path.join(
        cfg.model.checkpoint_path, "vocabs"
    )
    cfg.model.data_config.esm2_mappings_path = os.path.join(
        cfg.model.checkpoint_path, "vocabs"
    )

    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(cfg)
    model = instantiate(
        cfg.model,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )
    model.eval()

    state_dict = torch.load(
        cfg.model.inference_config.load_checkpoint, weights_only=True
    )
    model.load_state_dict(state_dict)

    if cfg.model.inference_config.pretrained_embedding is not None:
        if not isinstance(cfg.model.inference_config.pretrained_embedding, list):
            pretrained_embedding_paths = [
                cfg.model.inference_config.pretrained_embedding
            ]
        else:
            pretrained_embedding_paths = cfg.model.inference_config.pretrained_embedding
        model, gene_vocab = change_embedding_layer(model, pretrained_embedding_paths)
    return model, gene_vocab, gene2ensembl_id


def get_gene_similarity(
    gene_perturbed: str,
    gene_monitored: str,
    gene2ensembl_id: dict,
    model,
    gene_vocab: dict,
) -> float:
    from torch import nn

    def get_ensembl_id(gene: str) -> str:
        if gene in gene2ensembl_id:
            return str(np.random.choice(gene2ensembl_id[gene]))
        return "[PAD]"

    gene_perturbed_ensembl_id = get_ensembl_id(gene_perturbed)
    gene_monitored_ensembl_id = get_ensembl_id(gene_monitored)

    gene_perturbed_index = gene_vocab.get(
        gene_perturbed_ensembl_id, gene_vocab["[PAD]"]
    )
    gene_monitored_index = gene_vocab.get(
        gene_monitored_ensembl_id, gene_vocab["[PAD]"]
    )

    gene_perturbed_index = torch.tensor([gene_perturbed_index]).long()
    gene_monitored_index = torch.tensor([gene_monitored_index]).long()

    gene_embs = model.gene_embeddings.embedding
    gene_perturbed_emb = gene_embs(gene_perturbed_index)
    gene_monitored_emb = gene_embs(gene_monitored_index)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    gene_similarity = cos(gene_perturbed_emb, gene_monitored_emb)
    return float((gene_similarity[0].item() + 1)) / 2.0


@click.command()
@click.option(
    "--dataset-path",
    required=True,
    help="Path to the input dataset CSV file",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output-path",
    required=True,
    help="Path where to save the annotated dataset",
    type=click.Path(dir_okay=False),
)
def main(
    dataset_path: os.PathLike,
    output_path: os.PathLike,
):
    # Load dataset
    dataset_df = pd.read_csv(dataset_path)

    # Load model and vocab
    model, gene_vocab, gene2ensembl_id = load_transcriptformer()

    # Compute soft verifier scores
    probabilities = []
    for idx, row in dataset_df.iterrows():
        gene_perturbed = row["gene_perturbed"]
        gene_monitored = row["gene_monitored"]
        prob = get_gene_similarity(
            gene_perturbed, gene_monitored, gene2ensembl_id, model, gene_vocab
        )
        # Map cosine similarity [-1, 1] to [0, 1]
        prob = (prob + 1) / 2
        probabilities.append(prob)

    # Update dataframe
    dataset_df["class_confidences"] = [
        f"{1-prob:.4f}|{prob:.4f}" for prob in probabilities
    ]
    dataset_df["label"] = [int(prob > 0.5) for prob in probabilities]
    dataset_df["classes"] = "no|yes"

    dataset_df.to_csv(output_path, index=False)
    print(f"Annotated dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
