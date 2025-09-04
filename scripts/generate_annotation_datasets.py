# Usage: python generate_annotation_dataset.py --h5ad_filename tsv1 --predict_label cell_type --multiple-choice True

import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

from czi.ai.rbio.utils.utils import read_deepseek_system_prompt, read_template_prompt

RND_SEED = 42


def compute_class_confidences(x):
    classes = x["classes"].split("|")
    gt_class = x["label"]
    class_confidences = []
    for cl in classes:
        class_confidences.append(str(int(cl == gt_class)))
    return "|".join(class_confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotation Datasets Processing from h5ad to HF Datasets"
    )
    parser.add_argument(
        "--h5ad_filename",
        type=str,
        help="Name of the h5ad we want to process. One of: tsv1, tsv2_kidney, alzheimer, myeloid_cancer",
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
        default="v0.0.6",
    )
    parser.add_argument(
        "--predict_label",
        type=str,
        help="Label to predict. For tsv1, tsv2: cell_type; for alzheimer, myeloid_cancer: disease",
        default="cell_type",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory under which to save the files",
        default="/mnt/czi-sci-ai/project-rbio-40t/datasets/",
    )

    parser.add_argument(
        "--multiple-choice",
        type=bool,
        help="True if to include multi-class choices for predict_label in the prompt",
        default=False,
    )

    args = parser.parse_args()

    adata = sc.read_h5ad(f"{args.data_dir}/{args.h5ad_filename}")
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

    # user_prompt
    if args.predict_label != "cell_type":
        args.predict_label = args.predict_label + " state"
    template_prompt = read_template_prompt("annotation")
    adata_df["user_prompt"] = adata_df["gene_names"].apply(
        lambda x: template_prompt.replace("{0}", str(num_genes))
        .replace("{1}", x)
        .replace("{2}", args.predict_label)
    )
    adata_df["system_prompt"] = "You are an AI model trained as a Biologist through reinforcement learning. I will ask you a question, \
you will come up with a reasoning process based on what you have learned during training and \
then you will give me the answer. You will consider information learned during training about transcription factors, \
gene regulatory networks and gene co-expression and interaction patterns. \
The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You will provide the reasoning step-by-step, using detailed biological knowledge from training."
    # read_deepseek_system_prompt()
    adata_df["dataset_name"] = args.h5ad_filename
    adata_df["task"] = f"{args.predict_label}_prediction"
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

    # adata_df_train, adata_df_test = train_test_split(
    #     adata_df, test_size=1.0, random_state=RND_SEED
    # )
    # train_classes = adata_df_train["label"].unique()
    adata_df_test = adata_df.copy()
    test_classes = adata_df_test["label"].unique()

    # adata_df_train["classes"] = "|".join(train_classes)
    adata_df_test["classes"] = "|".join(test_classes)

    # adata_df_train["class_confidences"] = adata_df_train.apply(
    #     compute_class_confidences, 1
    # )
    adata_df_test["class_confidences"] = adata_df_test.apply(
        compute_class_confidences, 1
    )

    if args.multiple_choice:
        # adata_df_train["user_prompt"] = adata_df_train.apply(
        #     lambda x: x["user_prompt"]
        #     + f" The answer is one of: {' | '.join(x['classes'].split('|'))}",
        #     1,
        # )
        adata_df_test["user_prompt"] = adata_df_test.apply(
            lambda x: x["user_prompt"]
            + f" The answer is one of: {' | '.join(x['classes'].split('|'))}",
            1,
        )

    # train_save_filepath = f"{args.h5ad_filename.split('.h5ad')[0]}_top_{num_genes}_genes-train-{args.dataset_version}-multiple-choice-{args.multiple_choice}.csv"
    test_save_filepath = f"{args.h5ad_filename.split('.h5ad')[0]}_top_{num_genes}_genes-test-{args.dataset_version}-multiple-choice-{args.multiple_choice}-v0.2.0-system_prompt_self_aware_extra_CoT3.csv"

    # adata_df_train.to_csv(f"{args.data_dir}{train_save_filepath}", index=False)
    adata_df_test.to_csv(f"{args.data_dir}{test_save_filepath}", index=False)
    print(f"Succcess!")
    print("=" * 40)
    print(f"Saved adata files under {args.data_dir}")
    # print(f"\t adata_train: {train_save_filepath}")
    print(f"\t adata_test: {test_save_filepath}")
