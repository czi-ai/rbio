import os
import click
import random

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers.integrations import MLflowCallback
from trl import GRPOConfig, GRPOTrainer
from czi.ai.rbio.model.rewards import (
    composite_formatting_reward,
    genes_mentioned_in_think,
)
from czi.ai.rbio.utils.utils import extract_answer


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
        }

        yield return_data


class Reward:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.count = 0

    def compute_reward(
        self,
        completions: list,
        label: list,
        gene_perturbed: list,
        gene_monitored: list,
        system_prompt: list,
        user_promot: list,
        **kwargs,
    ):
        scores = []

        if self.count % 10 == 0:
            for completion, lbl, gp, gm, sys_p, usr_p in zip(
                completions,
                label,
                gene_perturbed,
                gene_monitored,
                system_prompt,
                user_promot,
            ):
                print(f"system prompt: {sys_p}")
                print(f"user prompt: {usr_p}")
                print(f"completion: {completion}")
                print(f"label: {(lbl == 1)}")
                print(f"gene perturbed: {gp}")
                print(f"gene monitored: {gm}")

        self.count += 1

        for completion, lbl, gp, gm, sys_p, usr_p in zip(
            completions,
            label,
            gene_perturbed,
            gene_monitored,
            system_prompt,
            user_promot,
        ):
            format_reward = composite_formatting_reward(completion)

            answer_from_text = extract_answer(completion)

            mention_reward = genes_mentioned_in_think(completion, gp, gm)

            bool_label = lbl == 1

            if answer_from_text is not None:
                answer_reward = float(answer_from_text == bool_label)
            else:
                answer_reward = 0

            total_score = format_reward + 2.0 * answer_reward + mention_reward

            scores.append(total_score)

        return scores


def train_fn(
    dataset_path: os.PathLike,
    model_name: str,
    output_dir: os.PathLike,
    resume_from_checkpoint: bool = False,
    trainer_args: GRPOConfig = None,
    per_device_train_batch_size: int = 4,
    num_generations: int = 4,
):
    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "false"
    os.environ["MLFLOW_TRACKING_URI"] = (
        "http://mlflow-api.mlflow.svc.cluster.local:5000"
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "rbio"

    df = pd.read_csv(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    dataset = Dataset.from_generator(
        dataset_gen, gen_kwargs={"dataset": df, "tokenizer": tokenizer}
    )

    if trainer_args is None:
        trainer_args = GRPOConfig(
            output_dir=str(output_dir),
            logging_steps=250,
            logging_first_step=True,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
        )

    trainer_args.output_dir = str(output_dir)

    reward = Reward(model, tokenizer)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward.compute_reward,
        args=trainer_args,
        train_dataset=dataset,
        callbacks=[MLflowCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


@click.command()
@click.option("--dataset-path", help="Dataset CSV file path", required=True)
@click.option(
    "--model-name", help="The name of the LLM model in huggingface", required=True
)
@click.option(
    "--checkpoint-dir", help="Directory where we save our checkpoints", required=True
)
@click.option(
    "--resume",
    help="Whether to resume from one of the checkpoints or not",
    default=False,
)
@click.option("--batch-size", help="Batch-size", default=4)
@click.option("--n-generations", help="Number of generations for GRPO", default=4)
def train(
    dataset_path: os.PathLike,
    model_name: str,
    checkpoint_dir: os.PathLike,
    resume: bool,
    batch_size: int,
    n_generations: int,
):
    train_fn(
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        per_device_train_batch_size=batch_size,
        num_generations=n_generations,
    )


if __name__ == "__main__":
    train()
