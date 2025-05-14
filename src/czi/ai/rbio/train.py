import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import click
import mlflow
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import MLflowCallback
from trl import GRPOConfig, GRPOTrainer

from czi.ai.rbio.model.rewards import (
    composite_formatting_reward,
    genes_mentioned_in_think,
    reward_answer_against_label,
    reward_answer_against_softverifier,
    reward_gene_similarity_via_vcm,
)
from czi.ai.rbio.model.verifiers import instantiate_vcm


def dataset_gen(dataset, tokenizer, balance_pos_neg=True):
    dataset_len = dataset.shape[0]
    df_true = dataset
    df_false = dataset

    if balance_pos_neg:
        df_true = dataset[dataset.label == 1]
        df_false = dataset[dataset.label == 0]

        dataset_len = max([len(df_true), len(df_false)]) * 2

    for i in range(dataset_len):
        if balance_pos_neg:
            if random.random() > 0.5:
                j = random.randint(0, df_true.shape[0] - 1)
                dataset_row = df_true.iloc[j]
            else:
                j = random.randint(0, df_false.shape[0] - 1)
                dataset_row = df_false.iloc[j]
        else:
            dataset_row = dataset.iloc[i]

        messages = [
            {"role": "system", "content": dataset_row["system_prompt"]},
            {"role": "user", "content": dataset_row["user_prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return_data = {
            "prompt": prompt,
            "label": dataset_row["label"],
            "gene_perturbed": dataset_row["gene_perturbed"],
            "gene_monitored": dataset_row["gene_monitored"],
            "task": dataset_row["task"],
            "system_prompt": dataset_row["system_prompt"],
            "user_prompt": dataset_row["user_prompt"],
        }

        yield return_data


class Reward:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        verifier_type: Optional[str] = "hard",
        vcm_verifier_type: Optional[str] = "transcriptformer",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.count = 0
        self.vcm_verifier_type = vcm_verifier_type
        self.verifier_type = verifier_type

        self.vcm_model = None
        self.vcm_gene_vocab = None
        self.gene2ensembl_id = None

    def init_vcm_model(self):
        self.vcm_model, self.vcm_gene_vocab, self.gene2ensembl_id = instantiate_vcm(
            self.vcm_verifier_type
        )

    def compute_reward(
        self,
        completions: list,
        label: list,
        gene_perturbed: list,
        gene_monitored: list,
        system_prompt: list,
        user_prompt: list,
        task: list,
        **kwargs,
    ):
        scores = []

        for completion, lbl, gp, gm, sys_p, usr_p, tsk in zip(
            completions,
            label,
            gene_perturbed,
            gene_monitored,
            system_prompt,
            user_prompt,
            task,
        ):
            format_reward = composite_formatting_reward(completion)

            mention_reward = genes_mentioned_in_think(completion, gp, gm)

            # reasoning_advantage_reward = compute_reasoning_advantage(
            #    self.model, self.tokenizer, sys_p, usr_p, completion, label
            # )

            reasoning_advantage_reward = 0

            if self.verifier_type == "hard":
                if tsk == "differential_expression":
                    answer_reward = reward_answer_against_label(completion, lbl == 1)
                elif tsk == "direction_of_change":
                    pass
            if self.verifier_type == "mlp":
                if tsk == "differential_expression":
                    answer_reward = reward_answer_against_softverifier(
                        completion, gp, gm
                    )
                elif tsk == "direction_of_change":
                    pass
            else:
                if tsk == "differential_expression":
                    if self.vcm_model is None:
                        self.init_vcm_model()  # lazy instantiation of vcm model

                    answer_reward = reward_gene_similarity_via_vcm(
                        gene_perturbed=gp,
                        gene_monitored=gm,
                        completion=completion,
                        task=tsk,
                        gene2ensembl_id=self.gene2ensembl_id,
                        vcm_model=self.vcm_model,
                        gene_vocab=self.vcm_gene_vocab,
                    )

            if self.count % 10 == 0:
                print(f"system prompt: {sys_p}")
                print(f"user prompt: {usr_p}")
                print(f"completion: {completion}")
                print(f"label: {(lbl == 1)}")
                print(f"gene perturbed: {gp}")
                print(f"gene monitored: {gm}")
                print(f"format reward: {format_reward}")
                print(f"mention reward: {mention_reward}")
                print(f"answer reward: {answer_reward}")
                print(f"reasoning advantage: {reasoning_advantage_reward}")

            total_score = (
                format_reward
                + 2.0 * answer_reward
                + mention_reward
                + reasoning_advantage_reward
            )
            # mlflow.log_metric("format_reward", format_reward, step=self.count)
            # mlflow.log_metric("mention_reward", mention_reward, step=self.count)
            # mlflow.log_metric("answer_reward", answer_reward, step=self.count)
            # mlflow.log_metric("reasoning_adv_reward", reasoning_advantage_reward, step=self.count)
            # mlflow.log_metric("total_score", total_score, step=self.count)

            scores.append(total_score)

        self.count += 1

        return scores


@dataclass
class RbioGRPOConfig(GRPOConfig):
    """
    This class extends GRPOConfig to add some parameters for logging to mlflow.
    Starting a run with mlflow and logging the parameters using code creates duplicates for every gpu.
    This is a workaround to avoid that.
    """

    model_name: Optional[str] = field(default=None)
    datasets: Optional[List[str]] = field(default=None)
    verifier_type: Optional[str] = field(default=None)
    batch_size: Optional[int] = field(default=None)


def train_fn(
    dataset_path: Union[os.PathLike, List[os.PathLike]],
    model_name: str,
    output_dir: os.PathLike,
    resume_from_checkpoint: bool = False,
    trainer_args: GRPOConfig = None,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    verifier_type: str = "hard",
):
    mlflow_run_name = os.environ.get(
        "MLFLOW_RUN_NAME",
        f"{model_name}_{verifier_type}_verifier_{num_generations}_generations_{per_device_train_batch_size}_batch_size",
    )

    if hasattr(dataset_path, "__iter__"):
        df_list = []

        for dp in dataset_path:
            df_list.append(pd.read_csv(dp))

        df = pd.concat(df_list)
    else:
        df = pd.read_csv(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    dataset = Dataset.from_generator(
        dataset_gen, gen_kwargs={"dataset": df, "tokenizer": tokenizer}
    )

    if trainer_args is None:
        trainer_args = RbioGRPOConfig(
            output_dir=str(output_dir),
            logging_steps=250,
            logging_first_step=True,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
            run_name=mlflow_run_name,
            datasets=dataset_path,
            model_name=model_name,
            verifier_type=verifier_type,
            batch_size=per_device_train_batch_size,
        )

    trainer_args.output_dir = str(output_dir)

    reward = Reward(model, tokenizer, verifier_type=verifier_type)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward.compute_reward,
        args=trainer_args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


# /mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/
@click.command()
@click.option(
    "--dataset-path",
    help="Dataset CSV file path",
    required=True,
    multiple=True,
    default=[
        "/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/hepg2-train-v0.1.1-no-augmentation.csv",
        "/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/jurkat-train-v0.1.1-no-augmentation.csv",
        "/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-train-v0.1.1-no-augmentation.csv",
        "/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/rpe1-train-v0.1.1-no-augmentation.csv",
    ],
)
@click.option(
    "--model-name",
    help="The name of the LLM model in huggingface",
    required=True,
    default="Qwen/Qwen2.5-3B-Instruct",
)
@click.option(
    "--checkpoint-dir",
    help="Directory where we save our checkpoints",
    required=True,
    default="/mnt/czi-sci-ai/project-rbio-large/checkpoints/PertQA-DE/All_Data/1_Rewrite/",
)
@click.option(
    "--resume",
    help="Whether to resume from one of the checkpoints or not",
    default=False,
)
@click.option("--batch-size", help="Batch-size", default=4)
@click.option("--n-generations", help="Number of generations for GRPO", default=4)
@click.option(
    "--verifier-type", help="type of verifier, hard, mlp or soft", default="hard"
)
def train(
    dataset_path: Union[os.PathLike, List[os.PathLike]],
    model_name: str,
    checkpoint_dir: os.PathLike,
    resume: bool,
    batch_size: int,
    n_generations: int,
    verifier_type: str,
):
    train_fn(
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        per_device_train_batch_size=batch_size,
        num_generations=n_generations,
        verifier_type=verifier_type,
    )


if __name__ == "__main__":
    train()
