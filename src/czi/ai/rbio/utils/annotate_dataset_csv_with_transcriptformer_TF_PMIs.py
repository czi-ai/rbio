import os
import torch
import pickle
import numpy as np
import pandas as pd
import click
from czi.ai.rbio.utils.utils import compute_soft_class_confidences, compute_binary_class_confidences, SYSTEM_PROMPT

TF_GENE_PMIS = os.getenv(
    "TF_GENE_PMIS",
    "/mnt/czi-sci-ai/project-rbio-40t/ana/rbio/datasets/transcriptformer/gene_pmis.pt",
)
TF_GENE2IDX = os.getenv(
    "TF_GENE2IDX",
    "/mnt/czi-sci-ai/project-rbio-40t/ana/rbio/datasets/transcriptformer/gene2idx.pkl",
)

TF_SIGNIFICANT_GENE_PAIRS = os.getenv(
    "TF_SIGNIFICANT_GENE_PAIRS",
    "/mnt/czi-sci-ai/project-rbio-40t/ana/rbio/datasets/transcriptformer/significant_gene_pairs_0.01.pkl",
)

TF_LEAST_SIGNIFICANT_GENE_PAIRS = os.getenv(
    "TF_LEAST_SIGNIFICANT_GENE_PAIRS",
    "/mnt/czi-sci-ai/project-rbio-40t/ana/rbio/datasets/transcriptformer/least_significant_gene_pairs_0.01.pkl",
)

def top_k_indices(array, k, reverse = True):
    flat_array = array.flatten()
    indices = np.argpartition(flat_array, -k)[-k:]
    indices = indices[np.argsort(flat_array[indices])]
    if reverse:
        indices = indices[:k]
    else:
        indices = indices[::-1]  # Sort indices by value
    return np.unravel_index(indices, array.shape)

def create_pmis_df(gene_indicesA, gene_indicesB, label):
    dataset_sig_df = pd.DataFrame({'gene_perturbed' : gene_indicesA, 
                               'gene_monitored' : gene_indicesB})
    dataset_sig_df['user_prompt'] = dataset_sig_df.apply(lambda x: f"Are gene {x['gene_perturbed']} and gene {x['gene_monitored']} likely to be co-expressed together? Give a binary yes/no answer only.", 1)
    dataset_sig_df['label'] = label 
    return dataset_sig_df


@click.command()
@click.option(
    "--binary-class-confidences",
    required=True,
    help="True if to annotate with binary class_confidences. False if to annotate with soft class_confidences",
    type=bool,
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
    output_dir: os.PathLike,
):
    # Load PMI matrix
    pmis = torch.load(TF_GENE_PMIS)
    pmis = pmis.clamp(min=0).numpy()
    
    # Load TF Geneidx vocab
    gene_vocab = pickle.load(open(TF_GENE2IDX, 'rb'))
    idx2gene = {v:k for k, v in gene_vocab.items()}
    pmis = (pmis - pmis.min()) / (pmis.max() - pmis.min())
    
    # Signficance threshold for gene pairs - we assume the top 0.01 pairs will be significant
    p_significant = 0.01

    num_genes = pmis.shape[0]
    num_gene_pairs = (num_genes * (num_genes - 1)) / 2
    topk = int(num_gene_pairs * p_significant)
    print(f'We believe {topk}, or {p_significant}% of gene pairs out of {num_gene_pairs} gene pairs would be significant interactions')
    
    # Retrieve the topk significant pairs
    topk_gene_indices1, topk_gene_indices2 = top_k_indices(pmis, topk, reverse = False)
    topk_gene_indices1_last, topk_gene_indices2_last = top_k_indices(pmis, topk, reverse=True)
    significant_gene_pairs = {(idx2gene[gene_A_idx], idx2gene[gene_B_idx]) : pmis[gene_A_idx, gene_B_idx] for gene_A_idx, gene_B_idx in zip(topk_gene_indices1, topk_gene_indices2)}
    least_significant_gene_pairs = {(idx2gene[gene_A_idx], idx2gene[gene_B_idx]) : pmis[gene_A_idx, gene_B_idx] for gene_A_idx, gene_B_idx in zip(topk_gene_indices1_last, topk_gene_indices2_last)}
    
    # Save topk significant pairs 
    with open(TF_SIGNIFICANT_GENE_PAIRS, "wb") as f:
        pickle.dump(significant_gene_pairs, f)
        
    # Save least topk significant pairs
    with open(TF_LEAST_SIGNIFICANT_GENE_PAIRS, "wb") as f:
        pickle.dump(least_significant_gene_pairs, f)
        
    # Compute cutoff threshold for PMIs
    cutoff = pmis[topk_gene_indices1, topk_gene_indices2].min()
    gene_indicesA = [idx2gene[gene_A_idx] for gene_A_idx in topk_gene_indices1]
    gene_indicesB = [idx2gene[gene_B_idx] for gene_B_idx in topk_gene_indices2]

    gene_indicesA_least = [idx2gene[gene_A_idx] for gene_A_idx in topk_gene_indices1_last]
    gene_indicesB_least = [idx2gene[gene_B_idx] for gene_B_idx in topk_gene_indices2_last]
    genes = gene_vocab.keys()
    print(f"Minimum PMI cutoff for significance of gene pairs is {cutoff}")
    
    # Dataset with positive labels
    dataset_df_sig = create_pmis_df(gene_indicesA, gene_indicesB, label = 'yes')
    
    # Dataset with negative labels
    dataset_df_least_sig = create_pmis_df(gene_indicesA_least, gene_indicesB_least, label = 'no')

    # Combine topk and least topk significant labels
    dataset_df = pd.concat([dataset_df_sig, dataset_df_least_sig])
    dataset_df['classes'] = 'yes|no'
    dataset_df['keywords'] = dataset_df.apply(lambda x: '|'.join([x['gene_perturbed'], x['gene_monitored']]), 1)
    dataset_df['system_prompt'] = SYSTEM_PROMPT
    dataset_df['cell_line'] = 'tf_3'
    dataset_df['dataset_name'] = 'TF_predictions'
    dataset_df['task'] = 'soft_verification'
    
    # Generate class confidences and save datasets
    if binary_class_confidences:
        dataset_df['class_confidences'] = dataset_df.apply(compute_binary_class_confidences, 1)
        output_filepath = f'{output_dir}/TF_PMIs_sig_{p_significant}-train-v0.0.3_binary.csv'
    else:
        gene_pairs2pmi = significant_gene_pairs
        gene_pairs2pmi.update(least_significant_gene_pairs)
        dataset_df['class_confidences'] = dataset_df.apply(lambda x: compute_soft_class_confidences(x ,gene_pairs2pmi) , 1)
        output_filepath = f'{output_dir}/TF_PMIs_sig_{p_significant}-train-v0.0.3_soft.csv'
    dataset_df.to_csv(output_filepath, index = False)
    print(f'Successs! Saved file to {output_filepath}!')

if __name__ == "__main__":
    main()