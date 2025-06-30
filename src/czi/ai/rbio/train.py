import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import click
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from czi.ai.rbio.model.rewards import (
    composite_formatting_reward,
    keywords_mentioned_in_think,
    reward_answer_against_go_ontology,
    reward_answer_against_label,
)
from czi.ai.rbio.model.verifiers import (
    instantiate_go_ontologies,
    instantiate_rouge_scorer,
)
from czi.ai.rbio.utils.metrics_collector import MetricsCollector
from datasets import Dataset


def differential_expression_dataset_generator(dataset, tokenizer, balance_pos_neg=True):
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
        verifier_type: Optional[Union[str, List[str]]] = "hard",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.count = 0

        self.verifiers = verifier_type[0]  # this is assumed to be a single term for now

        self.gene2ensembl_id = None

        self.metrics_collector = MetricsCollector()

        self.gene2go_annotations = None

        self.go_verifier = None

        self.go_rouge_scorer = None

        # Exponential Moving Average (EMA) normalization parameters
        self.reward_ema_norm_alpha = 0.001  # decay for stability
        self.reward_ema_mean = -np.inf
        self.reward_ema_var = 1.0
        self.epsilon = 1e-7

        self.use_go_ontology_verifier, self.go_ontology_type = (
            self.check_go_ontology_verifier(self.verifiers)
        )

    def check_go_ontology_verifier(self, verifier):
        """
        Check if there is a GO Ontology verifier in the list of verifiers

        Args:
            verifier: name of soft verifier; if GO, should follow the GO_{go_ontology_type}_{verifier_type}
                      i.e.: GO_L_rouge, GO_L_llh, GO_L_discrete

        Return:
            use_go_ontology: True if using a GO Ontology verifier
            go_verifier: the type of GO Verifier to use; one of: ['discrete', 'rouge', 'llh']
        """
        use_go_ontology = False
        go_verifier = []
        if verifier.startswith("GO"):
            use_go_ontology = True
            go_verifier = verifier.split("GO_")[1].split("_")[0]
        return use_go_ontology, go_verifier

    def normalize_reward_ema(self, reward):
        """
        Normalizes a reward using an Exponential Moving Average (EMA), a technique
        used to compute moving averages of rewards in an online fashion during RL training

        Args:
            reward: reward to be normalized

        Return:
            normalized reward

        """
        # Reward Mean and Variance get updated during training
        self.reward_ema_mean = (
            1 - self.reward_ema_norm_alpha
        ) * self.reward_ema_mean + self.reward_ema_norm_alpha * reward

        self.reward_ema_var = (
            1 - self.reward_ema_norm_alpha
        ) * self.reward_ema_var + self.reward_ema_norm_alpha * (
            reward - self.reward_ema_mean
        ) ** 2
        ema_std = (self.reward_ema_var + self.epsilon) ** 0.5

        # Reward gets normalized using updated mean and std
        norm_reward = (reward - self.reward_ema_mean) / ema_std

        # Map to [0, 1] interval by passing through the sigmoid fn
        norm_reward = 1 / (1 + np.exp(-norm_reward))
        return norm_reward

    def init_go_ontologies(self, go_ontology_type):
        """
        Initialize GO Ontology dictionary and ROUGE Scorer

        Args:
            go_ontology_type: type of GO Ontology to use - one of:
                F: GO Molecular Function
                C: GO Cellular Component
                P: Go Biological Process
        """
        self.gene2go_annotations = instantiate_go_ontologies(go_ontology_type)
        self.go_rouge_scorer = instantiate_rouge_scorer()

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
            format_reward = composite_formatting_reward(cmplt, self.go_verifier != None)

            mention_reward = keywords_mentioned_in_think(cmplt, kw)

            # reasoning_advantage_reward = compute_reasoning_advantage(
            #    self.model, self.tokenizer, sys_p, usr_p, completion, label
            # )

            reasoning_advantage_reward = 0
            answer_reward = 0

            if self.use_go_ontology_verifier:
                if self.gene2go_annotations is None:
                    self.init_go_ontologies(self.go_ontology_type)
                    self.go_verifier = self.verifiers.split("_")[2].split("_")[0]
                kw_genes = kw.split("|")
                answer_reward = reward_answer_against_go_ontology(
                    cmplt,
                    clss,
                    conf,
                    kw_genes,
                    self.go_verifier,
                    self.go_ontology_type,
                    self.gene2go_annotations,
                    self.go_rouge_scorer,
                    self.model,
                    self.tokenizer,
                )
                # initialize moving average with initial reward
                if self.reward_ema_mean == -np.inf:
                    self.reward_ema_mean = answer_reward
                answer_reward = self.normalize_reward_ema(answer_reward)
            else:
                answer_reward = reward_answer_against_label(cmplt, clss, conf)

            if self.count % 10 == 0:
                print(f"task: {tsk}")
                print(f"system prompt: {sys_p}")
                print(f"user prompt: {usr_p}")
                print(f"completion: {cmplt}")
                print(f"classes: {clss}")
                print(f"confidences per class: {conf}")
                print(f"label: {lbl}")
                print(f"keyworkds: {kw}")
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

            metrics = {
                "format_reward": format_reward,
                "mention_reward": mention_reward,
                "answer_reward": answer_reward,
                "reasoning_adv_reward": reasoning_advantage_reward,
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
    verifier_type: Union[str, List[str]] = "hard",
    trainer_args: Optional[RbioGRPOConfig] = None,
):
    mlflow_run_name = os.environ.get(
        "MLFLOW_RUN_NAME",
        f"{model_name}_{'-'.join(verifier_type)}_verifier_{num_generations}_generations_{per_device_train_batch_size}_batch_size",
    )
    print(mlflow_run_name)

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
        gen_kwargs={"dataset": df, "tokenizer": tokenizer},
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
            save_steps=10000,
            max_steps=100000,
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
    "--verifier-type",
    help="type of verifier, hard, mlp or soft",
    default=["GO_F_llh"],
    multiple=True,
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
