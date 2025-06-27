import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

from czi.ai.rbio.utils.utils import compute_binary_class_confidences

RND_SEED = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotation Datasets Processing from h5ad to HF Datasets"
    )
    parser.add_argument(
        "--h5ad_filename",
        type=str,
        help="Name of the h5ad we want to process",
        default="tsv1.h5ad",
    )
    parser.add_argument(
        "--gene_field",
        type=str,
        help="Field to use from adata.var for gene names",
        default="feature_name",
    )
    parser.add_argument(
        "--topk_genes",
        type=int,
        help="Number of top K genes to include during NL prompts",
        default=100,
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Version of the generated data files",
        default="v0.0.3",
    )
    parser.add_argument(
        "--predict_label", type=str, help="Label to predict", default="cell_type"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory under which to save the files",
        default="/mnt/czi-sci-ai/project-rbio-40t/datasets/",
    )

    args = parser.parse_args()

    adata = sc.read_h5ad(args.h5ad_filename)
    X = adata.X.toarray()
    num_genes = args.topk_genes
    num_obs = X.shape[0]
    gene_names = np.array(adata.var[args.gene_field].to_list())

    genes_sampled_indices = np.argsort(-X)[:, :num_genes]

    X_genes = []

    for i, gene_indices in enumerate(genes_sampled_indices):
        genes = gene_names[gene_indices]
        X_genes.append(", ".join(genes))

    # h5ad processed df containing the same information as in the adata file
    adata_df = pd.DataFrame(
        {
            "gene_names": X_genes,
            "cell_type": adata.obs["cell_type"].to_list(),
            "disease": adata.obs["disease"].to_list(),
            "assay": adata.obs["assay"].to_list(),
            "organism": adata.obs["organism"].to_list(),
            "tissue": adata.obs["tissue"].to_list(),
            "sex": adata.obs["sex"].to_list(),
            "dev_stage": adata.obs["development_stage"].to_list(),
            "self_rep_ethnicity": adata.obs["self_reported_ethnicity"].to_list(),
        }
    )

    adata_df["label"] = adata_df[args.predict_label]

    # system_prompt
    system_prompt = "A conversation between User and Biologist. The user asks a question, and the Biologist solves it. The biologist first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

    # user_prompt
    adata_df["user_prompt"] = adata_df["gene_names"].apply(
        lambda x: f"These are the top {num_genes} expressed genes in a single-cell observation belonging to a specific cell type: {x}. Based on these highly expressed genes, what do you predict the {args.predict_label} of this observation to be? Place the predicted {args.predict_label} under the <answer> </answer> tags and the reasoning process under the <think> </think> tags."
    )
    adata_df["system_prompt"] = system_prompt
    adata_df["dataset_name"] = args.h5ad_filename
    adata_df["task"] = f"{args.predict_label}_annotation"
    adata_df = adata_df[
        [
            "user_prompt",
            "system_prompt",
            "label",
            "task",
            "dataset_name",
            "disease",
            "assay",
            "organism",
            "tissue",
            "sex",
            "dev_stage",
            "self_rep_ethnicity",
        ]
    ]
    adata_df["keywords"] = ""

    adata_df_train, adata_df_test = train_test_split(
        adata_df, test_size=0.2, random_state=RND_SEED
    )
    train_classes = adata_df_train["label"].unique()
    test_classes = adata_df_test["label"].unique()

    adata_df_train["classes"] = "|".join(train_classes)
    adata_df_test["classes"] = "|".join(test_classes)

    adata_df_train["class_confidences"] = adata_df_train.apply(
        compute_class_confidences, 1
    )
    adata_df_test["class_confidences"] = adata_df_test.apply(
        compute_class_confidences, 1
    )

    train_save_filepath = f"{args.h5ad_filename.split('.h5ad')[0]}_top_{num_genes}_genes-train-{args.dataset_version}.csv"
    test_save_filepath = f"{args.h5ad_filename.split('.h5ad')[0]}_top_{num_genes}_genes-test-{args.dataset_version}.csv"

    adata_df_train.to_csv(f"{args.save_dir}{train_save_filepath}", index=False)
    adata_df_test.to_csv(f"{args.save_dir}{test_save_filepath}", index=False)
    print(f"Succcess!")
    print("=" * 40)
    print(f"Saved adata files under {args.save_dir}")
    print(f"\t adata_train: {train_save_filepath}")
    print(f"\t adata_test: {test_save_filepath}")
