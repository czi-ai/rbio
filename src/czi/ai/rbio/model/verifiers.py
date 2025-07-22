import json
import logging
import os
import pickle

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from rouge_score import rouge_scorer
from torch import nn

TF_CFG = os.getenv(
    "TF_CFG",
    "/mnt/czi-sci-ai/project-rbio-40t/transcriptformer/inference_config.yaml",
)
TF_MODEL_CKPT = os.getenv(
    "TF_MODEL_CKPT",
    "/mnt/czi-sci-ai/project-rbio-40t/transcriptformer/tf_sapiens",
)
GENE2ENSEMBL_ID_FILEPATH = os.getenv(
    "GENE2ENSEMBL_ID_FILEPATH",
    "/mnt/czi-sci-ai/project-rbio-40t/transcriptformer/gene2ensembl_ids.pkl",
)

GO_ONTOLOGIES_FILEPATH = os.getenv(
    "GO_ONTOLOGIES_FILEPATH", "/mnt/czi-sci-ai/project-rbio-40t/datasets/GO_Ontology/"
)


def instantiate_go_ontologies(go_ontology_type):
    """
    Instantiate GO Ontology dictionary based on the ontology type

    Args:
        go_ontology_type: type of GO Ontology to use - one of:
                F: GO Molecular Function
                C: GO Cellular Component
                P: Go Biological Process
                all: combine all together
    Return:
        gene2annotation: mapping from a gene to a list of annotations, based on the given GO ontology type
    """
    if go_ontology_type != "all":
        filepath = f"{GO_ONTOLOGIES_FILEPATH}gene_ontology_{go_ontology_type}.csv"
        gene2annotation = read_go_df(filepath)
    else:
        gene2annotation_c = read_go_df(f"{GO_ONTOLOGIES_FILEPATH}gene_ontology_C.csv")
        gene2annotation_p = read_go_df(f"{GO_ONTOLOGIES_FILEPATH}gene_ontology_P.csv")
        gene2annotation_f = read_go_df(f"{GO_ONTOLOGIES_FILEPATH}gene_ontology_F.csv")
        for g, ann in gene2annotation_p.items():
            if g in gene2annotation_c:
                gene2annotation_c[g] = gene2annotation_c[g] + ann
            else:
                gene2annotation_c[g] = ann
        for g, ann in gene2annotation_f.items():
            if g in gene2annotation_c:
                gene2annotation_c[g] = gene2annotation_c[g] + ann
            else:
                gene2annotation_c[g] = ann
        gene2annotation = gene2annotation_c
        print("annotation dict here")
        print(
            gene2annotation_c["CEBPB"],
            gene2annotation_f["CEBPB"],
            gene2annotation_p["CEBPB"],
            gene2annotation["CEBPB"],
        )
    return gene2annotation


def read_go_df(filepath):
    """
    Reads a GO ontology file from a filepath and converts it into a dictionary
    from genes to list of annotations

    Args:
        filepath: location of GO gene ontology

    Return:
        gene2annotation: mapping from a gene to a list of annotations, based on the given GO ontology type
    """
    go_df = pd.read_csv(filepath)
    go_df_grouped = go_df.groupby("gene").aggregate(list).reset_index()

    # combine all annotations for a given gene
    gene2annotation = dict(
        zip(
            go_df_grouped["gene"].to_list(),
            go_df_grouped["direct_class_label"].to_list(),
        )
    )
    return gene2annotation


def instantiate_rouge_scorer():
    """
    Instantiate a ROUGE Scorer
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer


def verify_gene_info_discrete(gene_info_llm, gene, gene2go_annotations):
    """
    Verify and reward gene information used by the reasoning model by lookup into a GO dictionary of annotations
    (established structured source of knowledge). The model gets a reward for each exact match for an annotation
    found in the answer.

    Args:
        gene_info_llm: gene information used by the llm, inside the <gene_info</<gene_info>
        gene: gene to retrieve annotations for
        gene2go_annotations: mapping from GO, from gene to a set of annotations

    Return:
        reward based on exact matches of annotations in the generated answer
    """
    reward = 0.0
    if gene not in gene2go_annotations:
        return 0.0
    # retrieve the annotations for that gene from GO
    gene_annotations = gene2go_annotations[gene]
    for gene_annotation in gene_annotations:
        # for each annotation from GO, if it's in the answer, give a rewad
        if gene_annotation in gene_info_llm:
            reward += 1
    # normalize by total number of annotations in GO
    return reward / len(gene_annotations)


def verify_gene_info_rouge_scores(gene_info_llm, gene, gene2go_annotations, scorer):
    """
    Verify and reward gene information used by the reasoning model by lookup into a GO dictionary of annotations
    (established structured source of knowledge). The model gets three rewards based on ROUGE scores between the annotations
    in the GO Ontology and the information in the <gene_info></gene_info>:
        rouge1: based on 1-gram overlap
        rouge1: based on 2-grams overlap
        rougeL: based on LCS (longest common subsequence)

    Args:
        gene_info_llm: gene information used by the llm, inside the <gene_info</<gene_info>
        gene: gene to retrieve annotations for
        gene2go_annotations: mapping from GO, from gene to a set of annotations
        scorer: ROUGE scorer to use; already instantiated

    Return:
        list of rewards: [rouge1_reward, rouge2_reward, rougeL_reward]
    """
    if gene not in gene2go_annotations:
        return 0.0, 0.0, 0.0
    gene_annotations = " ".join(gene2go_annotations[gene])
    scores = scorer.score(gene_info_llm, gene_annotations)
    rewards = [scores[key].fmeasure for key in ["rouge1", "rouge2", "rougeL"]]
    return rewards


def verify_gene_info_llh(gene, gene2go_annotations, model, tokenizer, go_ontology_type):
    """
    Verify and reward existing gene information about a given gene under the
    GO Ontology (established structured source of knowledge) by generating
    the log-likelihood of that information under the reasoning model that gets trained
    The model returns the log-likelihood.

    Args:
        gene: gene to retrieve annotations for
        gene2go_annotations: mapping from GO, from gene to a set of annotations
        model: RL model getting trained
        tokenizer: tokenizer corresponding to model
        go_ontology_type: type of ontology to use; one of F, C, P

    Return:
        log-likelihood of gene annotations for gene in the GO Ontology under the model
    """
    if gene not in gene2go_annotations:
        return 0.0
    # Combine gene annotations
    gene_annotations_combined = ", ".join(gene2go_annotations[gene])

    # Generate the ontology-specific prompts that will get evaluated under the model
    gene_annotation_c = f"Gene {gene} carries its molecular function in the following cellular components: {gene_annotations_combined}"
    gene_annotation_f = f"Gene {gene} or its gene products carry the following molecular-level activities inside a cell: {gene_annotations_combined}"
    gene_annotation_p = f"Gene {gene} is involved in the following biological processes: {gene_annotations_combined}"

    # Choose one prompt based on the type of ontology used
    gene_annotation = ""
    if go_ontology_type == "C":
        gene_annotation = gene_annotation_c
    elif go_ontology_type == "F":
        gene_annotation = gene_annotation_f
    elif go_ontology_type == "P":
        gene_annotation = gene_annotation_p
    else:
        gene_annotation = ". ".join(
            [gene_annotation_c, gene_annotation_f, gene_annotation_p]
        )

    # Tokenize gene_annotation
    input_ids = tokenizer.encode(gene_annotation, return_tensors="pt").long()

    # Pass the tokenized gene_annotation through the model
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        nll = outputs.loss.item()  # This gives back the avg NLL across tokens
    return -nll
