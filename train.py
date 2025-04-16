import os

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from utils.rewards import composite_formatting_reward, genes_mentioned_in_think
from utils.utils import extract_answer


def dataset_gen(dataset, tokenizer):
    for i in range(dataset.shape[0]):
        dataset_row = dataset.iloc[i]

        messages = [
            {"role": "system", "content": dataset_row['system_prompt']},
            {"role": "user", "content": dataset_row['user_prompt']},

        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return_data = {
            'prompt': prompt,
            'label': dataset_row['label'],
            'gene_perturbed': dataset_row['gene_perturbed'],
            'gene_monitored': dataset_row['gene_monitored'],
        }

        yield return_data

count = 0

def reward(completions, label, gene_perturbed, gene_monitored, **kwargs):
    scores = []

    global count
    if count % 10 == 0:
        for completion, lbl, gp, gm in zip(completions, label, gene_perturbed, gene_monitored):
            print(f'completion: {completion}')
            print(f'label: {(lbl == 1)}')
            print(f'gene perturbed: {gp}')
            print(f'gene monitored: {gm}')

    count += 1

    for completion, lbl, gp, gm in zip(completions, label, gene_perturbed, gene_monitored):
        format_reward = composite_formatting_reward(completion)

        answer_from_text = extract_answer(completion)

        mention_reward = genes_mentioned_in_think(completion, gp, gm)

        bool_label = (lbl == 1)

        if answer_from_text is not None:
            answer_reward = float(answer_from_text == bool_label)
        else:
            answer_reward = 0

        total_score = format_reward + answer_reward + mention_reward

        scores.append(total_score)

    return scores


def train(
        dataset_path: os.PathLike,
        model_name: str,
        output_dir: os.PathLike,
        resume_from_checkpoint: bool = False,
        trainer_args: GRPOConfig = None,
        per_device_train_batch_size: int = 4,
        num_generations: int = 4
):
    df = pd.read_csv(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = Dataset.from_generator(dataset_gen, gen_kwargs={'dataset': df, 'tokenizer': tokenizer})

    if trainer_args is None:
        trainer_args = GRPOConfig(
            output_dir=str(output_dir),
            logging_steps=10,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations
        )

    trainer_args.output_dir = str(output_dir)

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward,
        args=trainer_args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
