import os
import pathlib

import click
import pandas as pd
import perturbqa
from datasets import Dataset, load_dataset
from perturbqa import auc_per_gene, load_de, load_dir


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
        "classes": [],
        "class_confidences": [],
        "cell_line": [],
        "task": [],
        "dataset_name": [],
        "keywords": [],
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

            # Convert label to text and create confidence
            classes = "no|yes"
            class_confidences = "1.0|0.0" if label == 0 else "0.0|1.0"
            
            # Create keywords string
            keywords = f"{curr_data['pert']}|{curr_data['gene']}"

            dataset["system_prompt"].append(system_prompt)
            dataset["user_prompt"].append(question)
            dataset["label"].append(label)
            dataset["classes"].append(classes)
            dataset["class_confidences"].append(class_confidences)
            dataset["cell_line"].append(cell_line)
            dataset["task"].append("differential_expression")
            dataset["dataset_name"].append(perqa_dataset_name)
            dataset["keywords"].append(keywords)
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
    elif dataset_type == "direction_of_change" or "dir":
        pass
    else:
        raise NotImplementedError(f"dataset_type {dataset_type} not implemented")


if __name__ == "__main__":
    create_dataset()
