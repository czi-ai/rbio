import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import click
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from czi.ai.rbio.model.rewards import (
    composite_formatting_reward,
    keywords_mentioned_in_think,
    reward_answer_against_label,
)
from czi.ai.rbio.utils.metrics_collector import MetricsCollector


def differential_expression_dataset_generator(dataset, tokenizer, balance_pos_neg=True):
    dataset_len = dataset.shape[0]
    df_true = dataset
    df_false = dataset

    if balance_pos_neg:
        df_true = dataset[dataset.label != 0]
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
            "classes": dataset_row["classes"],
            "class_confidences": dataset_row["class_confidences"],
            "keywords": dataset_row["keywords"],
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
        answer_reward_on: bool = True,
        mention_reward_on: bool = True,
        format_reward_on: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.count = 0

        self.verifier_type = verifier_type

        self.gene2ensembl_id = None

        self.metrics_collector = MetricsCollector()

        self.answer_reward_on = answer_reward_on
        self.mention_reward_on = mention_reward_on
        self.format_reward_on = format_reward_on

    def compute_reward(
        self,
        completions: list,
        label: list,
        classes: list,
        class_confidences: list,
        keywords: list,
        system_prompt: list,
        user_prompt: list,
        task: list,
        **kwargs,
    ):
        scores = []
        metrics_batch = []

        for cmplt, lbl, clss, conf, kw, sys_p, usr_p, tsk in zip(
            completions,
            label,
            classes,
            class_confidences,
            keywords,
            system_prompt,
            user_prompt,
            task,
        ):
            if self.format_reward_on:
                format_reward = composite_formatting_reward(cmplt)
            else:
                format_reward = 0

            if self.mention_reward_on:
                mention_reward = keywords_mentioned_in_think(cmplt, kw)
            else:
                mention_reward = 0

            if self.answer_reward_on:
                answer_reward = reward_answer_against_label(cmplt, clss, conf)
            else:
                answer_reward = 0

            if self.count % 10 == 0:
                print(f"task: {tsk}")
                print(f"system prompt: {sys_p}")
                print(f"user prompt: {usr_p}")
                print(f"completion: {cmplt}")
                if self.format_reward_on:
                    print(f"format reward: {format_reward}")
                if self.mention_reward_on:
                    print(f"keyworkds: {keywords}")
                    print(f"mention reward: {mention_reward}")
                if self.answer_reward_on:
                    print(f"label: {lbl}")
                    print(f"answer reward: {answer_reward}")
                    print(f"classes: {clss}")
                    print(f"confidences per class: {conf}")

            total_score = format_reward + 2.0 * answer_reward + mention_reward

            metrics = {
                "format_reward": format_reward,
                "mention_reward": mention_reward,
                "answer_reward": answer_reward,
                "total_score": total_score,
            }
            metrics_batch.append(metrics)

            scores.append(total_score)

        self.metrics_collector.log_metrics(metrics_batch=metrics_batch, step=self.count)

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
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
    verifier_type: str = "hard",
    trainer_args: Optional[RbioGRPOConfig] = None,
    balance_pos_neg: bool = True,
    answer_reward_on: bool = True,
    mention_reward_on: bool = True,
    format_reward_on: bool = True,
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
        differential_expression_dataset_generator,
        gen_kwargs={
            "dataset": df,
            "tokenizer": tokenizer,
            "balance_pos_neg": balance_pos_neg,
        },
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
            save_steps=5000,
        )

    trainer_args.output_dir = str(output_dir)

    reward = Reward(
        model,
        tokenizer,
        verifier_type=verifier_type,
        answer_reward_on=answer_reward_on,
        mention_reward_on=mention_reward_on,
        format_reward_on=format_reward_on,
    )

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
@click.option(
    "--balance-pos-neg",
    help="Whether to balance the positive and negative examples",
    default=True,
)
@click.option(
    "--answer-reward-on",
    help="Whether to use answer reward",
    default=True,
)
@click.option(
    "--mention-reward-on",
    help="Whether to use mention reward",
    default=True,
)
@click.option(
    "--format-reward-on",
    help="Whether to use format reward",
    default=True,
)
def train(
    dataset_path: Union[os.PathLike, List[os.PathLike]],
    model_name: str,
    checkpoint_dir: os.PathLike,
    resume: bool,
    batch_size: int,
    n_generations: int,
    verifier_type: str,
    balance_pos_neg: bool,
    answer_reward_on: bool,
    mention_reward_on: bool,
    format_reward_on: bool,
):
    train_fn(
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        per_device_train_batch_size=batch_size,
        num_generations=n_generations,
        verifier_type=verifier_type,
        balance_pos_neg=balance_pos_neg,
        answer_reward_on=answer_reward_on,
        mention_reward_on=mention_reward_on,
        format_reward_on=format_reward_on,
    )


if __name__ == "__main__":
    train()
