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


def create_transcription_factors_df(transcription_factors, tf2interactions, label):
    """
    Creates dataframe based on pairs of transcription factors and a given label
    Annotates dataframe with a prompt.

    Args:
        transcription_factors: list of transcription factors
        tf2interactions: mapping of transcription factors to other transcription factors interactions
        label: 'yes' if interactions are known to be positive, 'no' otherwise

    Returns:
        annotated dataframe
    """
    dataset_sig_df = pd.DataFrame({"transcription_factor": transcription_factors})
    dataset_sig_df["gene_monitored"] = dataset_sig_df["transcription_factor"].apply(
        lambda x: tf2interactions[x]
    )
    dataset_sig_df = dataset_sig_df.explode("gene_monitored")
    dataset_sig_df["label"] = label
    return dataset_sig_df


@click.command()
@click.option(
    "--binary-class-confidences",
    required=True,
    help="True if to annotate with binary class_confidences. False if to annotate with soft class_confidences",
    type=bool,
)
@click.option(
    "--p_value",
    required=True,
    help="Significance level for p_value for gene interactions",
    type=float,
    default=0.05,
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
    p_value: float,
    min_not_sig_interactions_to_include: int,
    output_dir: os.PathLike,
):
    # TF GeneVocab
    gene_vocab = pkl.load(open(TF_GENE2IDX, "rb"))
    idx2gene = {v: k for k, v in gene_vocab.items()}

    # These values were precomputed ahead of time
    tfs = pkl.load(open(f"{TF_ROOT_DIR}p_values_tfs.pkl", "rb"))
    p_values = pkl.load(open(f"{TF_ROOT_DIR}p_values.pkl", "rb"))
    print(
        f"There are {len(tfs)} transcription factors and a total of {len(gene_vocab)} total genes in the vocab"
    )

    tf2sig_interactions = {}
    tf2not_sig_interactions = {}
    tf_pairs2p_vals = {}
    num_genes = p_values.shape[1]
    for tf_idx, tf in enumerate(tfs):
        tf_p_values = p_values[tf_idx]
        for gene_idx in range(num_genes):
            gene_name = idx2gene[gene_idx]
            tf_pairs2p_vals[(tf, gene_name)] = tf_p_values[gene_idx]
        tf2sig_interactions[tf] = [
            idx2gene[x[0]] for x in np.argwhere(tf_p_values <= p_value)
        ]

        not_sig_interactions_indices = np.argwhere(tf_p_values > p_value).reshape(
            -1,
        )
        num_samples = min(
            len(not_sig_interactions_indices), min_not_sig_interactions_to_include
        )
        not_sig_interactions_indices_sampled = np.random.choice(
            not_sig_interactions_indices, num_samples, replace=False
        )
        tf2not_sig_interactions[tf] = [
            idx2gene[x]
            for x in np.random.choice(
                not_sig_interactions_indices, num_samples, replace=False
            )
        ]

    # Significant interactions
    dataset_sig_df = create_transcription_factors_df(tfs, tf2sig_interactions, "yes")

    # Not Significant interactions
    dataset_not_sig_df = create_transcription_factors_df(
        tfs, tf2not_sig_interactions, "no"
    )

    # Combine significan and non-significant interactions
    dataset_df = pd.concat([dataset_sig_df, dataset_not_sig_df])
    dataset_df["user_prompt"] = dataset_df.apply(
        lambda x: f"If transcription factor {x['transcription_factor']} is activated, is expression of gene {x['gene_monitored']} going to be high? The answer is either yes or no.",
        1,
    )
    dataset_df["classes"] = "yes|no"
    dataset_df["keywords"] = dataset_df.apply(
        lambda x: "|".join([x["transcription_factor"], x["gene_monitored"]]), 1
    )
    dataset_df["system_prompt"] = SYSTEM_PROMPT
    dataset_df["cell_line"] = "tf_3"
    dataset_df["dataset_name"] = "TF_predictions"
    dataset_df["task"] = "soft_verification"
    dataset_df["label"] = dataset_df["label"].apply(lambda x: int(x == 'yes'))

    # Generate class confidences and save datasets
    if binary_class_confidences:
        dataset_df["class_confidences"] = dataset_df.apply(
            compute_binary_class_confidences, 1
        )
        output_filepath = (
            f"{output_dir}/TF_TFs2genes_p_{p_value}-train-v0.0.3_binary.csv"
        )
    else:
        dataset_df["class_confidences"] = dataset_df.apply(
            lambda x: compute_soft_class_confidences(
                x, tf_pairs2p_vals, ["transcription_factor", "gene_monitored"]
            ),
            1,
        )
        output_filepath = f"{output_dir}/TF_TFs2genes_p_{p_value}-train-v0.0.3_soft.csv"
    dataset_df.to_csv(output_filepath, index=False)
    print(f"Successs! Saved file to {output_filepath}!")


if __name__ == "__main__":
    main()
