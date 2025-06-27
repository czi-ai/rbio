import ast
import json
import os
import pickle as pkl

import click
import numpy as np
import pandas as pd

from czi.ai.rbio.utils.utils import (
    SYSTEM_PROMPT,
    compute_binary_class_confidences,
    compute_soft_class_confidences,
)

TF_ROOT_DIR = "/mnt/czi-sci-ai/project-rbio-40t/ana/rbio/datasets/transcriptformer/"

TF_GENE2IDX = os.getenv(
    "TF_GENE2IDX",
    f"{TF_ROOT_DIR}gene2idx.pkl",
)


def create_marker_genes_df(
    cell_types, cell_type2gene_name, cell_type2interactions, label
):
    """
    Creates dataframe based on pairs of cell types and transcription factors interactions and a given label
    Annotates dataframe with a prompt.

    Args:
        cell_types: list of cell_types
        cell_type2gene_name: mapping from cell_type2genes
        cell_type2interactions: mapping from cell_type2transcription_factors
        label: 'yes' if interactions are known to be positive, 'no' otherwise

    Returns:
        annotated dataframe
    """
    dataset_df = pd.DataFrame({"cell_type": cell_types})
    dataset_df["marker_genes"] = dataset_df["cell_type"].apply(
        lambda x: cell_type2gene_name[x]
    )
    dataset_df["gene_monitored"] = dataset_df["cell_type"].apply(
        lambda x: [y.strip() for y in cell_type2interactions[x]]
    )
    dataset_df["label"] = label
    return dataset_df


@click.command()
@click.option(
    "--binary-class-confidences",
    required=True,
    help="True if to annotate with binary class_confidences. False if to annotate with soft class_confidences",
    type=bool,
)
# This needs to be computed from the data, harcoded for now
@click.option(
    "--pmi_cutoff",
    required=True,
    help="Significance level for PMI scores",
    type=float,
    default=0.8254,
)
@click.option(
    "--min_not_sig_interactions_to_include",
    required=True,
    help="Number of minimum significant interactions to include per Transcription Factor",
    type=int,
    default=50,
)
@click.option(
    "--output-dir",
    required=True,
    help="Dir where to save the annotated dataset",
    type=click.Path(dir_okay=True),
    default="/mnt/czi-sci-ai/project-rbio-40t/datasets",
)
def main(
    binary_class_confidences: bool,
    pmi_cutoff: float,
    min_not_sig_interactions_to_include: int,
    output_dir: os.PathLike,
):
    cell_type2ensembl_id = json.load(open(f"{TF_ROOT_DIR}tsv2_marker_genes.json", "rb"))
    ensembl_id2gene_name = pkl.load(
        open(f"{TF_ROOT_DIR}ensembl_id2gene_name.pkl", "rb")
    )
    cell_type2gene_name = {
        ct: [
            (
                ensembl_id2gene_name[ensembl_id]
                if ensembl_id in ensembl_id2gene_name
                else ensembl_id
            )
            for ensembl_id in ct_info["gene_ids"]
        ]
        for ct, ct_info in cell_type2ensembl_id.items()
    }

    idx = 0
    for ct, marker_genes in cell_type2gene_name.items():
        print(f"Cell Type {ct} has {len(marker_genes)} marker genes")
        idx += 1
        if idx > 10:
            continue

    # Load precomputed Transcriptformer cell_type2transcription_factor gene expresssion levels
    pmis = pkl.load(open(f"{TF_ROOT_DIR}celltype2TFs_expressions.pkl", "rb"))
    pmis = np.exp(pmis)

    # Normalize PMIs
    pmis = (pmis - pmis.min()) / (pmis.max() - pmis.min())

    filtered_cell_types = pkl.load(open(f"{TF_ROOT_DIR}sorted_cell_types.pkl", "rb"))
    sorted_tfs = pkl.load(open(f"{TF_ROOT_DIR}sorted_tfs.pkl", "rb"))

    # Cell Type to significant TFs
    cell_type2significant_TFs = {}
    cell_type2less_significant_TFs = {}
    sorted_tfs = np.array(sorted_tfs)

    cell_type2tf_pmi = {}

    for cell_type_idx, cell_type in enumerate(filtered_cell_types):
        cell_type_pmis = pmis[cell_type_idx]
        sig_cell_type_pmis = sorted_tfs[
            np.argwhere(cell_type_pmis > pmi_cutoff)
        ].squeeze()
        not_sig_type_pmis = sorted_tfs[
            np.argwhere(cell_type_pmis <= pmi_cutoff)
        ].squeeze()

        for tf, tf_pmi in zip(sorted_tfs, cell_type_pmis):
            cell_type2tf_pmi[(cell_type, tf)] = tf_pmi

        if len(not_sig_type_pmis) > min_not_sig_interactions_to_include:
            not_sig_type_pmis = np.random.choice(
                not_sig_type_pmis, min_not_sig_interactions_to_include, replace=False
            )
        cell_type2significant_TFs[cell_type] = sig_cell_type_pmis
        cell_type2less_significant_TFs[cell_type] = not_sig_type_pmis

    print("Examples of significant TFs for cell_type cardiac endothelial cell: ")
    print("=" * 40)
    print(cell_type2significant_TFs["cardiac endothelial cell"])

    print("Examples of not significant TFs for cell_type cardiac endothelial cell: ")
    print("=" * 40)
    print(cell_type2less_significant_TFs["cardiac endothelial cell"])

    # Significant TFs interactions based on Marker Genes
    df_sig = create_marker_genes_df(
        filtered_cell_types, cell_type2gene_name, cell_type2significant_TFs, "yes"
    )

    # Not Significant TFs interactions based on Marker Genes
    df_not_sig = create_marker_genes_df(
        filtered_cell_types, cell_type2gene_name, cell_type2less_significant_TFs, "no"
    )

    # Combine significant and non significant interactions
    dataset_df = pd.concat([df_sig, df_not_sig])
    dataset_df = dataset_df.explode("gene_monitored")
    dataset_df["user_prompt"] = dataset_df.apply(
        lambda x: f"In a cell that has marker genes {', '.join(x['marker_genes'])} expressed, is transcription factor {x['gene_monitored']} likely to be activated? The answer is either yes or no.",
        1,
    )
    dataset_df["system_prompt"] = SYSTEM_PROMPT
    dataset_df["classes"] = "yes|no"
    dataset_df["keywords"] = dataset_df["gene_monitored"]
    dataset_df["cell_line"] = "tf_3"
    dataset_df["dataset_name"] = "TF_predictions"
    dataset_df["task"] = "soft_verification"
    dataset_df = dataset_df.drop(columns=["marker_genes"])

    dataset_df["label"] = dataset_df["label"].values.tolist()

    # Generate class confidences and save datasets
    if binary_class_confidences:
        dataset_df["class_confidences"] = dataset_df.apply(
            compute_binary_class_confidences, 1
        )
        output_filepath = (
            f"{output_dir}/TF_MGs2TFs_pmi_{pmi_cutoff}-train-v0.0.3_binary.csv"
        )
    else:
        dataset_df["class_confidences"] = dataset_df.apply(
            lambda x: compute_soft_class_confidences(
                x, cell_type2tf_pmi, ["cell_type", "gene_monitored"]
            ),
            1,
        )
        output_filepath = (
            f"{output_dir}/TF_MGs2TFs_pmi_{pmi_cutoff}-train-v0.0.3_soft.csv"
        )
    dataset_df.to_csv(output_filepath, index=False)
    print(f"Successs! Saved file to {output_filepath}!")


if __name__ == "__main__":
    main()
