import os
import click
import numpy as np
import pickle as pkl

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.integrations import MLflowCallback
from transformers.pipelines.pt_utils import KeyDataset
from trl import GRPOConfig, GRPOTrainer
from utils.rewards import composite_formatting_reward, genes_mentioned_in_think, vcm_reward_func
from utils.utils import extract_answer
from utils.verifiers import instantiate_vcm
from transformers import pipeline
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import re
os.environ['MLFLOW_TRACKING_USERNAME'] = ''
os.environ['MLFLOW_TRACKING_PASSWORD'] = ''

TRACKING_URI = "http://mlflow-api.mlflow.svc.cluster.local:5000"
EXP_NAME = "rbio"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXP_NAME)

GENE2ENSEMBL_ID_FILEPATH = 'transcriptformer/gene2ensembl_ids.pkl'
GENE2ENSEMBL_ID = pkl.load(open(GENE2ENSEMBL_ID_FILEPATH, 'rb'))
TF_MODEL, GENE_VOCAB = instantiate_vcm('transcriptformer')

TRAIN_DATASET_PATH = '/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-train-latest.csv'
TEST_DATASET_PATH = '/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-test-latest.csv'
train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)[:1000]

model_name = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)


def extract_answer_from_completion(completion):
    answer_matches = re.findall(r'<answer>.*?</answer>', completion, re.DOTALL)
    if len(answer_matches) < 2:
        return 'no'
    else:
        answer_matches = answer_matches[-1]
    answer = answer_matches.split('<answer>')[1].split('</answer>')[0].strip()
    return answer.lower()

def evaluate(pipe, test_dataset):
    """
    Compute metrics for a set of predictions and ground truth. Prints metrics and logs to mlflow
    
    Args:
        preds: predictions
        true: ground truth
    """
    prompts = test_dataset['prompt']
    predictions = pipe(prompts)
    generated_texts = [pred[0]['generated_text'] for pred in predictions]
    gen_texts_df = pd.DataFrame({'prompt' : prompts, 'generated_text' : generated_texts})
    gen_texts_df.to_csv('generated_texts.csv')
    mlflow.log_artifact('generated_texts.csv')
    answers = [extract_answer_from_completion(gt) for gt in generated_texts]
    preds = [1 * (answer == 'yes') for answer in answers]
    true = test_dataset['label']
    
    accuracy = accuracy_score(true, preds)
    p, r, f1, support = precision_recall_fscore_support(true, preds, average='macro')

    print(f"Accuracy: {accuracy}, Precision: {p}, Recall: {r}, F1: {f1}")
    mlflow.log_metric("accuracy", accuracy, step=count)
    mlflow.log_metric("precision", p, step=count)
    mlflow.log_metric("recall", r, step=count)
    mlflow.log_metric("F1", f1, step=count)
    
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
train_dataset = Dataset.from_generator(dataset_gen, gen_kwargs={'dataset': train_df, 'tokenizer': tokenizer})
test_dataset = Dataset.from_generator(dataset_gen, gen_kwargs={'dataset': test_df, 'tokenizer': tokenizer})

def reward(completions, label, gene_perturbed, gene_monitored, **kwargs):
    # print('TF Model', TF_MODEL)
    # print('Gene vocab', GENE_VOCAB)
    scores = []
    format_reward = 0
    answer_reward = 0
    mention_reward = 0
    vcm_reward_gene_similarity = 0
    total_score = 0

    global count
    if count % 10 == 0:
        print('Count', count)
        for completion, lbl, gp, gm in zip(completions, label, gene_perturbed, gene_monitored):
            print(f'completion: {completion}')
            print(f'label: {(lbl == 1)}')
            print(f'gene perturbed: {gp}')
            print(f'gene monitored: {gm}')
    count += 1
    # early stopping for sanity checking

    for completion, lbl, gp, gm in zip(completions, label, gene_perturbed, gene_monitored):
        format_reward = composite_formatting_reward(completion)

        answer_from_text = extract_answer(completion)

        mention_reward = genes_mentioned_in_think(completion, gp, gm)
        
        vcm_reward_gene_similarity = vcm_reward_func(gp, gm, completion, 'gene_similarity', GENE2ENSEMBL_ID, TF_MODEL, GENE_VOCAB)

        bool_label = (lbl == 1)

        if answer_from_text is not None:
            answer_reward = float(answer_from_text == bool_label)
        else:
            answer_reward = 0

        total_score = format_reward + answer_reward + mention_reward + vcm_reward_gene_similarity
        print(format_reward, answer_reward, mention_reward, vcm_reward_gene_similarity, total_score)    

        scores.append(total_score)
    if count % 10 == 0:
        mlflow.log_metric("format_reward", format_reward, step=count)
        mlflow.log_metric("answer_reward", answer_reward, step=count)
        mlflow.log_metric("mention_reward", mention_reward, step=count)
        mlflow.log_metric("vcm_reward_gene_similarity", vcm_reward_gene_similarity, step=count)
        mlflow.log_metric("all_reward", total_score, step=count)

    return scores


def train_fn(
        train_dataset_path: os.PathLike,
        test_dataset_path: os.PathLike,
        model_name: str,
        output_dir: os.PathLike,
        resume_from_checkpoint: bool = False,
        trainer_args: GRPOConfig = None,
        per_device_train_batch_size: int = 4,
        num_generations: int = 4,
        vcm_model: str = 'transcriptformer',
        task: str = 'gene_similarity',
        batch_size: int = 4,
        n_generations: int = 4,
        dataset: str = 'K562'
):
    
    
    if trainer_args is None:
        trainer_args = GRPOConfig(
            output_dir=str(output_dir),
            logging_steps=10,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
            max_steps=100
        )

    trainer_args.output_dir = str(output_dir)

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward,
        args=trainer_args,
        train_dataset=train_dataset,
        callbacks=[MLflowCallback()],
    )
    run_name = f"rbio-{model_name}-vcm-{vcm_model}-{task}-{dataset}"
    
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    print(node_rank)
    if node_rank == 0:
        # mlflow.pytorch.autolog()
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("vcm_model", vcm_model)
        mlflow.log_param("reward_type", 'vcm_tf_gene_similarity')
        mlflow.log_param("task", task)
        mlflow.log_param("train_dataset", train_dataset_path)
        mlflow.log_param("test_dataset", test_dataset_path)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("n_generations", n_generations)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    pipe = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer, max_length=1024)
    evaluate(pipe, test_dataset)
    

@click.command()
@click.option('--train-dataset-path', help='Train Dataset CSV file path', required=True, default="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-train-latest.csv")
@click.option('--test-dataset-path', help='Test Dataset CSV file path', required=True, default="/mnt/czi-sci-ai/project-rbio/AutoSync/Datasets/PertQA-DE/k562-test-latest.csv")
@click.option('--model-name', help='The name of the LLM model in huggingface', required=True, default="Qwen/Qwen2.5-3B-Instruct")
@click.option('--checkpoint-dir', help='Directory where we save our checkpoints', required=True, default="checkpoints")
@click.option('--resume', help='Whether to resume from one of the checkpoints or not', default=False)
@click.option('--batch-size', help='Batch-size', default=4)
@click.option('--n-generations', help='Number of generations for GRPO', default=4)
@click.option('--vcm_model', help='Name of VCM model', default='transcriptformer')
@click.option('--task', help='task', default='gene_similarity')
@click.option('--dataset', help='dataset', default='K562')
@click.option('--reward_type', help='reward_type', default='soft-verifier-vcm')
def train(
        train_dataset_path: os.PathLike,
        test_dataset_path: os.PathLike,
        model_name: str,
        checkpoint_dir: os.PathLike,
        resume: bool,
        batch_size: int,
        n_generations: int,
        vcm_model: str, 
        task: str,
        dataset: str,
        reward_type: str
):
    
    
    train_fn(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        model_name=model_name,
        output_dir=checkpoint_dir,
        resume_from_checkpoint=resume,
        per_device_train_batch_size=batch_size,
        num_generations=n_generations,
        vcm_model=vcm_model,
        task=task,
        batch_size=batch_size,
        n_generations=n_generations,
        dataset=dataset
    ) 

if __name__ == '__main__':
    train()