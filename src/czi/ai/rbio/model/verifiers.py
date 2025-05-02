import json
import logging
import os
import sys

import anndata
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader

module_dir = os.path.join(
    os.path.dirname(__file__), "../transcriptformer/src", "transcriptformer"
)
sys.path.append(module_dir)

print(sys.path)

from data.dataloader import AnnDataset
from model.embedding_surgery import change_embedding_layer
from tf_utils.utils import stack_dict
from tokenizer.vocab import load_vocabs_and_embeddings

TF_CFG = "/opt/jupyter-envs/rbio/rbio-dev-ana/work/rbio/transcriptformer/conf/inference_config.yaml"
TF_MODEL_CKPT = "/opt/jupyter-envs/rbio/rbio-dev-ana/work/rbio/transcriptformer/checkpoints/tf_sapiens"


def call_vcm(
    gene_perturbed,
    gene_monitored,
    gene2ensembl_id,
    model,
    gene_vocab,
    verification_type="gene_similarity",
):
    gene_perturbed_ensembl_id = (
        str(np.random.choice(gene2ensembl_id[gene_perturbed]))
        if gene_perturbed in gene2ensembl_id
        else "[PAD]"
    )
    gene_monitored_ensembl_id = (
        str(np.random.choice(gene2ensembl_id[gene_monitored]))
        if gene_monitored in gene2ensembl_id
        else "[PAD]"
    )
    gene_perturbed_index = (
        gene_vocab[gene_perturbed_ensembl_id]
        if gene_perturbed_ensembl_id in gene_vocab
        else gene_vocab["[PAD]"]
    )
    gene_monitored_index = (
        gene_vocab[gene_monitored_ensembl_id]
        if gene_monitored_ensembl_id in gene_vocab
        else gene_vocab["[PAD]"]
    )

    gene_perturbed_index = torch.Tensor([gene_perturbed_index]).long()
    gene_monitored_index = torch.Tensor([gene_monitored_index]).long()
    if verification_type == "gene_similarity":
        gene_embs = model.gene_embeddings.embedding
        gene_perturbed_emb = gene_embs(gene_perturbed_index)
        gene_monitored_emb = gene_embs(gene_monitored_index)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        gene_similarity = cos(gene_perturbed_emb, gene_monitored_emb)
    elif verification_type == "mutual_information":
        return 0

    return gene_similarity[0]


def instantiate_vcm(model_type):
    if model_type == "transcriptformer":
        cfg = yaml.load(open(TF_CFG, "r"), Loader=yaml.SafeLoader)
        config_path = os.path.join(cfg["model"]["checkpoint_path"], "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        mlflow_cfg = OmegaConf.create(config_dict)

        # Merge the MLflow config with the main config
        cfg = OmegaConf.create(cfg)
        cfg = OmegaConf.merge(mlflow_cfg, cfg)

        # Set the checkpoint paths based on the unified checkpoint_path
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
        # print('Gene VOCAB', gene_vocab)

        # Instantiate the model
        logging.info("Instantiating the model")
        model = instantiate(
            cfg.model,
            gene_vocab_dict=gene_vocab,
            aux_vocab_dict=aux_vocab,
            emb_matrix=emb_matrix,
        )
        model.eval()
        logging.info("Model instantiated successfully")

        # Check if checkpoint is supplied
        if (
            not hasattr(cfg.model.inference_config, "load_checkpoint")
            or not cfg.model.inference_config.load_checkpoint
        ):
            raise ValueError(
                "No checkpoint provided for inference. Please specify a checkpoint path in "
                "model.inference_config.load_checkpoint"
            )

        logging.info("Loading model checkpoint")
        # Instead of loading full checkpoint, just load weights
        state_dict = torch.load(
            cfg.model.inference_config.load_checkpoint, weights_only=True
        )

        # Validate and load weights
        # converter.validate_loaded_weights(model, state_dict)
        model.load_state_dict(state_dict)
        logging.info("Model weights loaded successfully")

        # Perform embedding surgery if specified in config
        if cfg.model.inference_config.pretrained_embedding is not None:
            logging.info("Performing embedding surgery")
            # Check if pretrained_embedding_paths is a list, if not convert it to a list
            if not isinstance(cfg.model.inference_config.pretrained_embedding, list):
                pretrained_embedding_paths = [
                    cfg.model.inference_config.pretrained_embedding
                ]
            else:
                pretrained_embedding_paths = (
                    cfg.model.inference_config.pretrained_embedding
                )
            model, gene_vocab = change_embedding_layer(
                model, pretrained_embedding_paths
            )
        return model, gene_vocab


def check_math_solution(prompt, completion):
    """
    Verifier for math solutions. Checks if a completion corresponding to a math solution is correct.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a solution to a math equation
    Returns:
        True if the math solution corresponds to the given prompt
    """
    # TO-DO: complete implementation
    return True


def test_code_solution(prompt, completion):
    """
    Verifier for coding. Tests if completion corresponding to a coding task is correct.
    Runs the code and returns True if code runs.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a program
    Returns:
        True if the completion corresponding to the generated code runs
    """
    # TO-DO: complete implementation
    return True


def test_vcm_task(prompt, completion, vcm, task):
    """
    Verifier for VCM task. Tests if completion corresponding to a VCM task is correct.
    Runs the task using VCM and returns True if task is verifiable by the VCM model.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a program
        vcm: vcm model to use
    Returns:
        True if the completion is verifiable by the vcm
    """
    # TO-DO: complete implementation
    return True
