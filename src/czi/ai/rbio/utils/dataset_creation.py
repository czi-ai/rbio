import pathlib

import perturbqa
import os
import pandas as pd
from perturbqa import load_de, load_dir, auc_per_gene
from datasets import Dataset, load_dataset
import click


def generate_prompt_column(
    row, strict_binary=False, prompt_type="de_expression", cell_type="K562"
):
    """
    Generates a prompt columns

    Args:
        dataset_name: pertq_dataset name
    Returns:
        train_dataset: training_dataset
        test_dataset: test_dataset
        X_train_keys: training keys for pertqa_dataset
        X_test_keys: testing keys for pertqa_dataset
    """
    print(f"I am using a binary answer: {strict_binary} for the task {prompt_type}")
    gene_A = row["pert"]
    gene_B = row["gene"]

    if prompt_type == "gene_de_expression":
        row["prompt"] = (
            f"Is a knockdown of {gene_A} in {cell_type} cells likely to result in differential expression of {gene_B}?"
        )
    elif prompt_type == "gene_dir_change":
        row["prompt"] = (
            f"Is a knockdown of {gene_A} in {cell_type} cells likely to result in an increase of {gene_B}?"
        )
    if strict_binary:
        row["prompt"] += 'Give only a "Yes" or "No" answer.'
    # else:
    # row['prompt'] += 'At the end of your answer, give a Yes or No. Please provide your entire reasoning trace'
    return row


def generate_dataset_from_pertqa(dataset_name, logger, num_rows=-1):
    """
    Generates a HuggingFace dataset from a pertqa dataset. Returns the top num_rows from the dataset

    Args:
        dataset_name: pertq_dataset name
    Returns:
        train_dataset: training_dataset
        test_dataset: test_dataset
        X_train_keys: training keys for pertqa_dataset
        X_test_keys: testing keys for pertqa_dataset
    """
    data_de = load_de(dataset_name)
    X_train = data_de["train"]
    X_test = data_de["test"]
    logger.info(f"Total number of observations: {len(X_train)}")

    X_train_keys = [(x["pert"], x["gene"]) for x in X_train][:num_rows]
    X_test_keys = [(x["pert"], x["gene"]) for x in X_test][:num_rows]

    data_dir = load_dir("k562")
    train_dataset = Dataset.from_list(
        [{"pert": x["pert"], "gene": x["gene"], "label": x["label"]} for x in X_train]
    )
    test_dataset = Dataset.from_list(
        [{"pert": x["pert"], "gene": x["gene"], "label": x["label"]} for x in X_test]
    )
    if num_rows != -1:
        train_dataset = train_dataset.select(range(num_rows))
        test_dataset = test_dataset.select(range(num_rows))
    return train_dataset, test_dataset, X_train_keys, X_test_keys


def extract_dataset_from_pertqa(dataset_name: str, split: str):
    pertqa_path = os.environ.get("PERTQA_PATH", perturbqa.__path__[0])
    dataset_csv_path = os.path.join(pertqa_path, "datasets", dataset_name + ".csv")
    perturb_qa_data = pd.read_csv(dataset_csv_path)
    perturb_qa_filtered = perturb_qa_data[perturb_qa_data.split == split]

    return perturb_qa_filtered


def create_differential_expression_dataset_csv_dataset(
    perqa_dataset_name: str,
    cell_line: str,
    dataset_savepath: os.PathLike,
    split: str,
    system_prompt_path: os.PathLike = os.path.join(
        os.path.dirname(__file__),
        "templates/system_prompts/system_prompt_deepseek_adapted.txt",
    ),
    user_prompt_template_path: os.PathLike = os.path.join(
        os.path.dirname(__file__),
        "templates/differential_expression_prompt_single_template.txt",
    ),
):
    pertqa_dataset_filtered = extract_dataset_from_pertqa(perqa_dataset_name, split)

    dataset = {
        "system_prompt": [],
        "user_prompt": [],
        "label": [],
        "dataset_name": [],
        "gene_perturbed": [],
        "gene_monitored": [],
    }

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    with open(user_prompt_template_path, "r") as f:
        user_prompt_templates = [line.rstrip() for line in f]

    for i in range(len(pertqa_dataset_filtered)):
        for prompt_template in user_prompt_templates:
            direction, prompt_template = prompt_template.split(":")

            curr_data = pertqa_dataset_filtered.iloc[i]

            question = prompt_template.format(
                curr_data["pert"], cell_line, curr_data["gene"]
            )

            if direction == "D":
                label = curr_data["label"]
            elif direction == "R":
                label = 1 - curr_data["label"]

            dataset["system_prompt"].append(system_prompt)
            dataset["user_prompt"].append(question)
            dataset["label"].append(label)
            dataset["dataset_name"].append(perqa_dataset_name)
            dataset["gene_perturbed"].append(curr_data["pert"])
            dataset["gene_monitored"].append(curr_data["gene"])

    prompt_dataset = pd.DataFrame(dataset)
    prompt_dataset.to_csv(dataset_savepath, index=False)


def generate_dataset_from_norman_query(task):
    """
    Generates a HuggingFace dataset from a query on the Norman datset

    Args:
        task: type of task within the Norman dataset
    Returns:
        test_dataset: test dataset
    """
    prompts = []
    perts = []
    if task == "cell_cycle_position":
        pert_pairs = [
            (None, "CDKN1A"),
            (None, "CDKN1B"),
            ("CDKN1C", None),
            ("CDKN1C", "CDKN1A"),
            ("CDKN1C", "CDKN1B"),
            ("PLK4", None),
            (None, "STIL"),
            ("PLK4", "STIL"),
            ("CKS1B", None),
            ("KIF18B", None),
            (None, "KIF2C"),
            ("KIF18B", "KIF2C"),
        ]
        cell_cycle_positions = ["M", "M-G1", "G1-S", "S", "G2-M"]
        for gene_A, gene_B in pert_pairs:
            if gene_A and gene_B:
                # Would you expect it to be arrested at a particular stage?
                # If the answer is yes, then at what stage? make is multiple choice
                prompt = f"How would an overexpression of {gene_A} and {gene_B} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            elif gene_A:
                prompt = f"How would an overexpression of {gene_A} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            elif gene_B:
                prompt = f"How would an overexpression of {gene_B} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            prompt += (
                f"Choose one of the following cycles: {str(cell_cycle_positions)}?"
            )
            prompts.append({"prompt": prompt})
        print(prompts)
        test_dataset = Dataset.from_list(prompts)

    return test_dataset


@click.command()
@click.argument("dataset_type")
@click.option("--perqa-dataset-name", help="PertQA dataset name", required=True)
@click.option(
    "--cell-line", help="The name of the cell line to build prompt", required=True
)
@click.option(
    "--dataset-savepath", help="Directory where we save our dataset", required=True
)
@click.option(
    "--split",
    help="The name of the split",
)
@click.option(
    "--system-prompt-path",
    help="The path of the system prompt template",
    default=pathlib.Path(
        os.path.join(
            os.path.dirname(__file__),
            "templates/system_prompts/system_prompt_deepseek_adapted.txt",
        )
    ),
)
@click.option(
    "--user-prompt-path",
    help="The path of the user prompt template",
    default=pathlib.Path(
        os.path.join(
            os.path.dirname(__file__),
            "templates/differential_expression_prompt_single_template.txt",
        )
    ),
)
def create_dataset(
    dataset_type: str,
    perqa_dataset_name: str,
    cell_line: str,
    dataset_savepath: os.PathLike,
    split: str,
    system_prompt_path: os.PathLike,
    user_prompt_path: os.PathLike,
):
    if dataset_type == "differential_expression" or "de":
        create_differential_expression_dataset_csv_dataset(
            perqa_dataset_name,
            cell_line,
            dataset_savepath,
            split,
            system_prompt_path,
            user_prompt_path,
        )
    else:
        raise NotImplementedError(f"dataset_type {dataset_type} not implemented")


if __name__ == "__main__":
    create_dataset()
