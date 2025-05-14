import json
import logging
import os
import pickle

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from transcriptformer.model.embedding_surgery import change_embedding_layer
from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
import pandas as pd
from rouge_score import rouge_scorer

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

GO_ONTOLOGIES_FILEPATH = os.getenv(
    "GO_ONTOLOGIES_FILEPATH",
     "/mnt/czi-sci-ai/project-rbio-large/datasets/GO_Ontology/"
)

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

    gene_embs = model.gene_embeddings.embedding
    gene_perturbed_emb = gene_embs(gene_perturbed_index)
    gene_monitored_emb = gene_embs(gene_monitored_index)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    gene_similarity = cos(gene_perturbed_emb, gene_monitored_emb)

    return gene_similarity[0]


def instantiate_vcm(model_type):
    if model_type == "transcriptformer":
        gene2ensemble_id = pickle.load(open(GENE2ENSEMBL_ID_FILEPATH, "rb"))

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
        return model, gene_vocab, gene2ensemble_id

def instantiate_go_ontologies(go_ontology_type):
    if go_ontology_type != 'all':
        filepath = f'{GO_ONTOLOGIES_FILEPATH}gene_ontology_{go_ontology_type}.csv'
        gene2annotation = read_go_df(filepath)
    else:
        gene2annotation_c = read_go_df(f'{GO_ONTOLOGIES_FILEPATH}gene_ontology_C.csv')
        gene2annotation_p = read_go_df(f'{GO_ONTOLOGIES_FILEPATH}gene_ontology_P.csv')
        gene2annotation_f = read_go_df(f'{GO_ONTOLOGIES_FILEPATH}gene_ontology_F.csv')
        gene2annotation_c.update(gene2annotation_p)
        gene2annotation_c.update(gene2annotation_f)
        gene2annotation = gene2annotation_c
    return gene2annotation   

def instantiate_rouge_scorer():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer    
    
def read_go_df(filepath):
    go_df = pd.read_csv(filepath)
    go_df_grouped = go_df.groupby('gene').aggregate(list).reset_index()
    gene2annotation = dict(zip(go_df_grouped['gene'].to_list(), go_df_grouped['direct_class_label'].to_list()))
    return gene2annotation

    
def verify_gene_info(
    gene_info_llm, 
    gene, 
    gene2go_annotations):
    
    reward = 0.0
    if gene not in gene2go_annotations:
        return 0.0
    gene_annotations = gene2go_annotations[gene]
    for gene_annotation in gene_annotations:
        print(gene_annotation)
        if gene_annotation in gene_info_llm:
            reward += 1
    return reward / len(gene_annotations)


def verify_gene_info(
    gene_info_llm, 
    gene, 
    gene2go_annotations):
    
    reward = 0.0
    if gene not in gene2go_annotations:
        return 0.0
    gene_annotations = gene2go_annotations[gene]
    for gene_annotation in gene_annotations:
        # print(gene_annotation)
        if gene_annotation in gene_info_llm:
            reward += 1
    return reward / len(gene_annotations)


def verify_gene_info_rouge_scores(
    gene_info_llm, 
    gene, 
    gene2go_annotations, 
    scorer):
    
    if gene not in gene2go_annotations:
        return 0.0, 0.0, 0.0
    gene_annotations = ' '.join(gene2go_annotations[gene])
    scores = scorer.score(gene_info_llm, gene_annotations)
    rewards = [scores[key].fmeasure for key in ['rouge1', 'rouge1', 'rougeL']]     
    return rewards


def verify_gene_info_llh(
    gene, 
    gene2go_annotations, 
    model, 
    tokenizer, 
    go_ontology_type):
    print('go ontology type', go_ontology_type)
    if gene not in gene2go_annotations:
        return 0.0
    print('made it here', gene)
    gene_annotations_combined = ', '.join(gene2go_annotations[gene])
    gene_annotation_c = f'Gene {gene} carries its molecular function in the following cellular components: {gene_annotations_combined}'
    gene_annotation_f = f'Gene {gene} or its gene products carry the following molecular-level activities inside a cell: {gene_annotations_combined}'
    gene_annotation_p = f'Gene {gene} is involved in the following biological processes: {gene_annotations_combined}'
    
    gene_annotation = ""
    if go_ontology_type == 'C':
        gene_annotation = gene_annotation_c
    elif go_ontology_type == 'F':
        gene_annotation = gene_annotation_f
    elif go_ontology_type == 'P':
        gene_annotation = gene_annotation_p
    # print("gene_annotation", gene_annotation)
    input_ids = tokenizer.encode(gene_annotation, return_tensors = 'pt')
    input_ids = input_ids.to(model.device)
    # print(input_ids)
    # return 0.0
    # print(input_ids)                                                                                                   return 0.0
    with torch.no_grad():
        print('made it hereeeeee')
        print(input_ids)
        outputs = model(input_ids, labels=input_ids)
        # print(5454)
        nll = outputs.loss.item() #avg nll across tokens
        # print(nll)
    return -nll
    
    
    
    