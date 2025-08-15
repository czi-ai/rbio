import os
import re
from getpass import getpass

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import click
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint
from tqdm import tqdm



def download_sharded_checkpoint_from_s3(s3_client, s3_bucket, prefix, local_dir):
    """
    Downloads model weights from a AWS S3 bucket to a local directory.

    Args:
        s3_client: instantiation of s3 client to use
        s3_bucket: AWS S3 bucket where the weights are located
        s3_prefix: AWS S3 dir prefix for model weights location
        local_dir: directory where the model weights will be stored
    """
    os.makedirs(local_dir, exist_ok=True)

    # First, collect all files to download
    files_to_download = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            filename = os.path.basename(s3_key)
            if filename:  # Skip directories
                files_to_download.append((s3_key, filename, obj["Size"]))

    # Download with progress bar
    for s3_key, filename, file_size in tqdm(files_to_download, desc="Downloading files"):
        local_path = os.path.join(local_dir, filename)

        # Check if file already exists and has correct size
        if os.path.exists(local_path) and os.path.getsize(local_path) == file_size:
            print(f"  Skipping {filename} (already downloaded)")
            continue

        class ProgressCallback:
            def __init__(self, filename, file_size):
                self.filename = filename
                self.pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"  {filename}")

            def __call__(self, bytes_transferred):
                self.pbar.update(bytes_transferred)

            def close(self):
                self.pbar.close()

        callback = ProgressCallback(filename, file_size)
        try:
            s3_client.download_file(s3_bucket, s3_key, local_path, Callback=callback)
        finally:
            callback.close()


def load_model_and_tokenizer(model_name, model_checkpoint, device):
    """
    Loads a model and tokenizer from a trained checkpoint

    Args:
        model_name: name of base model to load
        model_checkpoint: name of rbio model_checkpoint to use
        device: device to use
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_checkpoint is not None:
        print(f"Loading model weights from {model_checkpoint}")
        load_sharded_checkpoint(model, model_checkpoint, strict=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.to(device)
    return model, tokenizer


def ask_rbio_multiple_questions(
    questions, device, system_prompt, system_prompt_type, model_ckpt, tokenizer
):
    """
      Queries rbio for multiple questions

      Args:
        questions: list of questions to ask rbio
        device: system_device to use
        system_prompt: system_prompt to use for the trained model
        system_prompt_type: type of system_prompt to use
        model_ckpt: model ckpt to use for inference
        tokenizer: tokenizer to use

    Returns:
        responses_df: pandas dataframe containing answers
    """
    answers = []
    think_traces = []
    reflections = []
    for question in questions:
        responses = ask_rbio_single_question(
            system_prompt, question, device, model_ckpt, tokenizer
        )
        answers.append(responses["answer"])
        think_traces.append(responses["think"])
        reflections.append(responses["reflection"])
    responses_df = pd.DataFrame(
        {
            "system_prompt_type": system_prompt_type * len(answers),
            "system_prompt": system_prompt * len(answers),
            "question": questions,
            "think_trace": think_traces,
            "reflection": reflections,
            "answer": answers,
        }
    )
    return responses_df


def ask_rbio_single_question(system_prompt, question, device, model_ckpt, tokenizer):
    """
    Queries rbio on a single question

    Args:
        system_prompt: system_prompt to use
        question: question to ask
        device: device to use
        model_ckpt: model ckpt to use for inference
        tokenizer: tokenizer to use

    Returns:
        responses: dictionary containing the thinking_trace, reflection, and final answers
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    # For more diversity in answers, we recommend tuning the temperature and top_p
    generated_ids = model_ckpt.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    think_process = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    reflection = re.findall(r"<reflect>(.*?)</reflect>", response, flags=re.DOTALL)
    answer = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)

    if len(think_process) > 0:
        think_process = think_process[0]
    if len(reflection) > 0:
        reflection = reflection[0]
    if len(answer) > 0:
        answer = answer[0]
    respones = {
        "think": think_process,  # thinking trace
        "reflection": reflection,  # reflection if model has one
        "answer": answer,
    }  # final answer
    return respones


def inference_fn(
    aws_s3_bucket,
    base_model_name,
    rbio_ckpt,
    system_prompt,
    system_prompt_type,
    questions,
    results_output_folder,
    results_output_filename,
):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    prefix = f"{rbio_ckpt}/"

    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    local_dir = f"model_weights/{rbio_ckpt}/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(local_dir):
        print(f"Downloading model weights from {aws_s3_bucket}/{prefix}")
        download_sharded_checkpoint_from_s3(s3, aws_s3_bucket, prefix, local_dir)

    model, tokenizer = load_model_and_tokenizer(base_model_name, local_dir, device)

    answers_df = ask_rbio_multiple_questions(
        questions, device, system_prompt, system_prompt_type, model, tokenizer
    )
    answers_df["model_type"] = rbio_ckpt

    if not os.path.exists(results_output_folder):
        os.makedirs(results_output_folder)
        print(f"Folder '{results_output_folder}' created successfully.")
    answers_df.to_csv(f"{results_output_folder}/{results_output_filename}")


@click.command()
@click.option(
    "--aws_s3_bucket",
    help="aws_s3_bucket for the model weights",
    default="czi-rbio",
)
@click.option(
    "--aws_s3_prefix",
    help="AWS_S3_BUCKET_PREFIX for the model weights",
    default="rbio_TF_ckpt",
)
@click.option(
    "--base_model_name", help="base model name", default="Qwen/Qwen2.5-3B-Instruct"
)
@click.option("--rbio_model_ckpt", help="rbio_model_ckpt", default="rbio_TF_ckpt")
@click.option(
    "--results_output_folder",
    help="optional folder where to save the results",
    default="predictions",
)
@click.option(
    "--results_output_filename",
    help="optional filename for the results",
    default="results.csv",
)
def run_rbio_inference(
    aws_s3_bucket: str,
    aws_s3_prefix: str,
    base_model_name: str,
    rbio_model_ckpt: str,
    results_output_folder: str,
    results_output_filename: str,
):
    # These can be changed
    CoT_suffix = " The Biologist provides the reasoning step-by-step."

    system_prompt_orig = "A conversation between User and Biologist. The user asks a question, \
    and the Biologist solves it. The biologist first thinks about the reasoning process in the mind and \
    then provides the user with the answer. The reasoning process and answer are enclosed within \
    <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

    system_prompt_orig_CoT = system_prompt_orig + CoT_suffix

    questions = [
        "Is a knockdown of CPAMD8  in rpe1 cells likely \
    to result in differential expression of SPARC? The answer is either yes or no. "
    ]

    inference_fn(
        aws_s3_bucket=aws_s3_bucket,
        base_model_name=base_model_name,
        rbio_ckpt=rbio_model_ckpt,
        system_prompt=system_prompt_orig_CoT,
        system_prompt_type="system_prompt_orig_CoT",
        questions=questions,
        results_output_folder=results_output_folder,
        results_output_filename=results_output_filename,
    )


if __name__ == "__main__":
    run_rbio_inference()
