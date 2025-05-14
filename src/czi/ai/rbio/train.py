import os
import ast
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
    has_cellular_component,
    has_localizes_mention,
    genes_mentioned_in_think,
    reward_answer_against_label,
    reward_gene_similarity_via_vcm,
    reward_gene_information_go_ontology_mention,
    reward_gene_information_go_ontology_rouge_score,
    reward_go_info_llh
)
from czi.ai.rbio.model.verifiers import instantiate_vcm, instantiate_go_ontologies, instantiate_rouge_scorer


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
        soft_verifiers: Optional[List] = ['go_ontology'],
        go_ontology_type: Optional[str] = 'c',
        verifier_type: Optional[str] = "hard",
        vcm_verifier_type: Optional[str] = "transcriptformer",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.count = 0
        self.vcm_verifier_type = vcm_verifier_type
        self.verifier_type = verifier_type
        self.go_ontology_type = go_ontology_type
        self.soft_verifiers = soft_verifiers

        self.vcm_model = None
        self.vcm_gene_vocab = None
        self.gene2ensembl_id = None
        self.gene2go_annotations = None
       

    def init_vcm_model(self):
        self.vcm_model, self.vcm_gene_vocab, self.gene2ensembl_id = instantiate_vcm(
            self.vcm_verifier_type
        )
        
    def init_go_ontologies(self):
        self.gene2go_annotations = instantiate_go_ontologies(self.go_ontology_type)
        self.rouge_scorer = instantiate_rouge_scorer()
        # print('Annotations dict', self.gene2go_annotations)

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
            
            has_cellular_component_reward = has_cellular_component(completion)
        
            has_localizes_mention_reward = has_localizes_mention(completion)

            # reasoning_advantage_reward = compute_reasoning_advantage(
            #    self.model, self.tokenizer, sys_p, usr_p, completion, label
            # )

            reasoning_advantage_reward = 0

            if self.verifier_type == "hard":
                if tsk == "differential_expression":
                    answer_reward = reward_answer_against_label(completion, lbl == 1)
                elif tsk == "direction_of_change":
                    pass
            elif self.verifier_type == "soft":
                if "gene_similarity" in self.soft_verifiers:
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
                if "go_ontology" in self.soft_verifiers:
                    # print('Made it here!')
                    if self.gene2go_annotations is None:
                        self.init_go_ontologies() 

                    go_reward_gp_discrete = reward_gene_information_go_ontology_mention(gp, completion, self.gene2go_annotations)
                    go_reward_gm_discrete = reward_gene_information_go_ontology_mention(gm, completion, self.gene2go_annotations)
                    
                    go_reward_gp_rouge1, go_reward_gp_rouge2, go_reward_gp_rougel = reward_gene_information_go_ontology_rouge_score(gp, completion, self.gene2go_annotations, self.rouge_scorer)
                    go_reward_gm_rouge1, go_reward_gm_rouge2, go_reward_gm_rougel = reward_gene_information_go_ontology_rouge_score(gm, completion, self.gene2go_annotations, self.rouge_scorer)
                    answer_reward = 0.0
                    go_info_llh_gp = reward_go_info_llh(gp, self.gene2go_annotations, self.model, self.tokenizer, self.go_ontology_type)
                    go_info_llh_gm = reward_go_info_llh(gm, self.gene2go_annotations, self.model, self.tokenizer, self.go_ontology_type)
                

            if self.count % 10 == 0:
                print(f"system prompt: {sys_p}")
                print(f"user prompt: {usr_p}")
                print(f"completion: {completion}")
                print(f"label: {(lbl == 1)}")
                print(f"gene perturbed: {gp}")
                print(f"gene monitored: {gm}")
                print(f"format reward: {format_reward}")
                print(f"mention reward: {mention_reward}")
                print(f"gene perturbed go ontology reward: {go_reward_gp_discrete}")
                print(f"gene monitored go ontology reward: {go_reward_gm_discrete}")
                print(f"gene perturbed go ontology reward_rouge_scores: {go_reward_gp_rouge1, go_reward_gp_rouge2, go_reward_gp_rougel}")
                print(f"gene monitored go ontology reward_rouge_scores: {go_reward_gm_rouge1, go_reward_gm_rouge2, go_reward_gm_rougel}")
                print(f"answer reward: {answer_reward}")
                print(f"reasoning advantage: {reasoning_advantage_reward}")
                print(f"has cellular component reward {has_cellular_component_reward}")
                print(f"has_localizes_mention_reward {has_localizes_mention_reward}")
                print(f"GO Info LLH GP {go_info_llh_gp}")
                print(f"GO Info LLH GM {go_info_llh_gm}")
                      

            total_score = (
                format_reward
                + 2.0 * answer_reward
                + mention_reward
                + reasoning_advantage_reward
                + go_reward_gp_discrete
                + go_reward_gm_discrete
                + go_reward_gp_rouge1 + go_reward_gp_rouge2 + go_reward_gp_rougel
                + go_reward_gm_rouge1 + go_reward_gm_rouge2 + go_reward_gm_rougel
                + has_cellular_component_reward 
                + has_localizes_mention_reward
                + go_info_llh_gp
                + go_info_llh_gm

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
    soft_verifiers: list = ['go_ontology'],
    go_ontology_type: str = 'c'
):
    mlflow_run_name = os.environ.get(
        "MLFLOW_RUN_NAME",
        f"{model_name}_{verifier_type}_verifier_{('').join(ast.literal_eval(soft_verifiers))}_GO_ontology_{go_ontology_type}_{num_generations}_generations_{per_device_train_batch_size}_batch_size",
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
            max_steps=3,  # this is for testing purposes; needs to be changed for full training
            run_name=mlflow_run_name,
            datasets=dataset_path,
            model_name=model_name,
            verifier_type=verifier_type,
            batch_size=per_device_train_batch_size,
        )

    trainer_args.output_dir = str(output_dir)

    reward = Reward(model, tokenizer, verifier_type=verifier_type, soft_verifiers=soft_verifiers, go_ontology_type=go_ontology_type)

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
        "/mnt/czi-sci-ai/project-rbio-large/datasets/hepg2-train-v0.1.6-go_ontology.csv",
        "/mnt/czi-sci-ai/project-rbio-large/datasets/jurkat-train-v0.1.6-go_ontology.csv",
        "/mnt/czi-sci-ai/project-rbio-large/datasets/k562-train-v0.1.6-go_ontology.csv",
        "/mnt/czi-sci-ai/project-rbio-large/datasets/rpe1-train-v0.1.6-go_ontology.csv",
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
@click.option(
    "--soft_verifiers",
    help="List of soft verifiers to use",
    default=['go_ontology'],
)
@click.option(
    "--go_ontology_type",
    help="Type of go ontology to use",
    default='C',
)
@click.option("--batch-size", help="Batch-size", default=4)
@click.option("--n-generations", help="Number of generations for GRPO", default=4)
@click.option("--verifier-type", help="type of verifier, hard or soft", default="soft")
def train(
    dataset_path: Union[os.PathLike, List[os.PathLike]],
    model_name: str,
    checkpoint_dir: os.PathLike,
    resume: bool,
    batch_size: int,
    n_generations: int,
    verifier_type: str,
    soft_verifiers: List[str],
    go_ontology_type: str
):
    train_fn(
        dataset_path=dataset_path,
        model_name=model_name,
        output_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        per_device_train_batch_size=batch_size,
        num_generations=n_generations,
        verifier_type=verifier_type,
        soft_verifiers=soft_verifiers,
        go_ontology_type=go_ontology_type
    )


if __name__ == "__main__":
    train()
