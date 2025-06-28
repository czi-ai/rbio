import hashlib
import json
import re

# General system prompt
SYSTEM_PROMPT = "A conversation between User and Biologist. The user asks a question, and the Biologist solves it. The biologist first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

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
