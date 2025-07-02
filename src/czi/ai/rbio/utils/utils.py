import hashlib
import json
import os
import re

import numpy as np

# Transcriptformer Directories and filepaths
TF_ROOT_DIR = "/mnt/czi-sci-ai/project-rbio-40t/transcriptformer/"

TF_GENE2IDX = os.getenv(
    "TF_GENE2IDX",
    f"{TF_ROOT_DIR}gene2idx.pkl",
)

TF_GENE_PMIS = os.getenv(
    "TF_GENE_PMIS",
    f"{TF_ROOT_DIR}gene_pmis.pt",
)

TF_SIGNIFICANT_GENE_PAIRS = os.getenv(
    "TF_SIGNIFICANT_GENE_PAIRS",
    f"{TF_ROOT_DIR}significant_gene_pairs_0.01.pkl",
)

TF_LEAST_SIGNIFICANT_GENE_PAIRS = os.getenv(
    "TF_LEAST_SIGNIFICANT_GENE_PAIRS",
    "{TF_ROOT_DIR}least_significant_gene_pairs_0.01.pkl",
)

# Directory of prompts
PROMPTS_DIR = "templates/system_prompts"


def read_template_prompt(prompt_type):
    """
    Returns a specific template for a given prompt_type
    """
    if prompt_type == "TF_PMIs":
        prompt = read_prompt("templates/TF_PMIs_prompt_template.txt")
    elif prompt_type == "TF_MGs2TF":
        prompt = read_prompt(
            "templates/TF_marker_genes2transcription_factor_prompt_template.txt"
        )
    elif prompt_type == "TF_TFs2TF":
        prompt = read_prompt(
            "templates/TF_transcription_factor2transcription_factor_prompt_template.txt"
        )
    return prompt.split("D: ")[1]


def read_deepseek_system_prompt():
    """
    Returns the system prompt used in DeepSeek
    """
    return read_prompt(f"{PROMPTS_DIR}/system_prompt_deepseek_adapted.txt")


def read_prompt(prompt_filepath):
    """
    Reads a prompt from a given txt file

    Args:
        prompt_filepath: location of the prompt, as txt file

    Return:
        content of the txt file
    """
    try:
        with open(prompt_filepath, "r") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file from {prompt_filepath} was not found.")
    except Exception as e:
        print(f"An error occurredin reading the prompt from {prompt_filepath}: {e}")
    return content


def normalize_scores(m, sig_threshold, reverse=False):
    """
    Normalizes a matrix m to the interval [0, 1] such that values at a given signifcance threshold
    sig_threshold correspond to values at 0.5 in the unit-normalized matrix m_norm

    No assumption are made on the range of values in m or on sig_threshold

    Args:
        m: matrix to normalize
        sig_threshold: significance threshold in
    """
    m_norm = np.zeros(m.shape)
    m_min = m.min()
    m_max = m.max()
    m_below_mask = m < sig_threshold
    m_above_mask = m >= sig_threshold
    m_below_norm = 0.5 * (m[m_below_mask] - m_min) / (sig_threshold - m_min)
    m_above_norm = 0.5 + 0.5 * (m[m_above_mask] - sig_threshold) / (
        m_max - sig_threshold
    )

    # Adjust for when values lower than sig_threshold should map to [0.5, 1],
    # rather than values > sig_threshold
    if reverse:
        m_above_norm -= 0.5
        m_below_norm += 0.5

    m_norm[m_below_mask] = m_below_norm
    m_norm[m_above_mask] = m_above_norm
    print(
        f"Matrix normalized. Values < {sig_threshold:0.4f} mapped to [{m_norm[m_below_mask].min():0.2f}, {m_norm[m_below_mask].max():0.2f}]. Values >= {sig_threshold:0.4f} mapped to [{m_norm[m_above_mask].min():0.2f}, {m_norm[m_above_mask].max():0.2f}] | m_norm_min: {m_norm.min():0.4f}. m_norm_max: {m_norm.max():0.4f}"
    )
    return m_norm


def compute_binary_class_confidences(x):
    """
    Computes binary class confidences corresponding to the CSV schema for soft verification
    datasets, given a series x.

    x is assumed to have the following fields:
     - classes: all class labels
     - label: the ground truth class label

    The function returns a string correponding to class_confidences for all the classes

    Example:
        classes = yes|no
        label = yes

    Function returns class_confidences = 1|0

    Args:
        x: dataframe
    """
    classes = x["classes"].split("|")
    gt_class = x["label"]
    class_confidences = []
    for cl in classes:
        class_confidences.append(str(int(cl == gt_class)))
    return "|".join(class_confidences)


def compute_soft_class_confidences(x, gene_pair2scores, fields):
    """
    Computes soft class confidences corresponding to the CSV schema for soft verification
    datasets, given a series x.

    x is assumed to have the following fields:
     - classes: all class labels
     - label: the ground truth class label
     - fields: list of two fields corresponding to a pair of genes

    The function returns a string correponding to class_confidences for all the classes for the pair of genes

    Example:
        fields[0]: gene_A
        fields[1]: gene_B
        classes = yes|no
        label = yes

    Function returns class_confidences = gene_pair2scores[(gene_A, gene_B)]|0

    Args:
        x: dataframe
        gene_pair2scores: dictionary corresponding to gene pairs and confidence interaction scores
    """
    gene_A = x[fields[0]]
    gene_B = x[fields[1]]
    pmi = gene_pair2scores[(gene_A, gene_B)]
    classes = x["classes"].split("|")
    gt_class = x["label"]
    class_confidences = []
    for cl in classes:
        class_confidences.append(str(int(cl == gt_class) * pmi))
    return "|".join(class_confidences)


def extract_binary_answer(text):
    found = re.search(r"<answer>\s*(yes|no)\s*</answer>", text, re.IGNORECASE)
    if found:
        return found.group(1).strip().lower()

    return ""


def extract_think(text, separator="\n"):
    think_contents = re.findall(
        r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
    )
    return separator.join(think_contents).strip()


def compute_embeddings_hash(emb_dict: dict) -> str:
    # Convert embeddings to a stable string representation
    emb_str = json.dumps({k: v.tolist() for k, v in sorted(emb_dict.items())})
    return hashlib.md5(emb_str.encode()).hexdigest()
