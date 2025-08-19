# # Training Rbio (A Demo)
# In this script we will demonstrate how to train Rbio on perturbation data obtained from
# the PertQA dataset available originally published here https://github.com/genentech/PerturbQA and adapted to our use case.
#
# Rbio implements LLM post-training using soft-verification mechanisms so that knowledge from biology models such as a virtual cell model (VCM) can be distilled and used within the LLM, rather than relying on hard ground truth labels obtained experimentally which are usually scarce and often costly.
#
# In this example, we use a simplified "VCM" consisting of a Multi Layer Perceptron (MLP) trained to answer perturbation questions. It exposes an interface that returns a probability when prompted with two gene names. This is the probability that a knockout of gene_a is having an effect on the expression of gene_b.
#
# We use this signal as a soft verification signal within our reward mechanism in order to post-train our LLM. This improves the LLM capabilities to answer questions of the form "Is a knockdown of <gene_a> in <cell_line> cells likely to result in differential expression of gene_b?"

# ## Imports, global variables, random seeds

import os
from typing import List

import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from trl import GRPOTrainer

from rewards import *

from utils import (
    set_random_seeds,
    load_mlp_classifier,
    setup_model_and_tokenizer,
    create_training_config,
    mlp_classifier_inference
)

# Disabling logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

# Training configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
N_STEPS = 100000
BATCH_SIZE = 4
NUM_GENERATIONS = 4
SAVE_EVERY = 10000
OUTPUT_DIR = "./checkpoints"

# Global step counter
STEP_COUNT = 0

# MLP classifier configuration (mlp was not trained on k562 cells)
MLP_MODEL_PATH = "./mlp_model.pt"
EMBEDDING_FILE = "./esm_embedding_dictionary_filled.pkl"

# Dataset paths
DATASET_PATHS = [
    "./k562-train-v0.3.0.csv",
]

# Set seeds globally
set_random_seeds(42)


# # Simplified VCM
# Our simplified virtual cell model (VCM) is a MLP Classifier. This serves the purpose of a soft labeler for our reward strategy, as it returns probabilities of gene pairs being differentially expressed.

# In[ ]:


class MLPClassifier(nn.Module):
    """Simple MLP classifier for gene pair classification"""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model(x)
        return result


# Global variables for MLP model and embeddings
mlp_model = None
embeddings_dict = None


# # Dataset
#
# `load_and_prepare_dataset` has the purpose of loading the training datasets (k562 only in this example) as a pandas dataframe.
#
# `create_mlp_labeled_dataset_generator` yields samples from this dataset that have been soft-labeled by our simplified VCM: the MLP defined above.

# In[ ]:


def load_and_prepare_dataset(dataset_paths: List[str], balance_pos_neg: bool = True) -> pd.DataFrame:
    """Load CSV datasets and combine them into a single DataFrame"""
    if len(dataset_paths) == 1:
        dataset_df = pd.read_csv(dataset_paths[0])
    else:
        dataset_list = []
        for path in dataset_paths:
            dataset_list.append(pd.read_csv(path))
        dataset_df = pd.concat(dataset_list, ignore_index=True)

    print(f"Loaded dataset with {len(dataset_df)} rows")
    return dataset_df


def create_mlp_labeled_dataset_generator(dataset_df: pd.DataFrame, tokenizer, balance_pos_neg: bool = True):
    """Generate training examples with MLP-based labeling"""
    if balance_pos_neg:
        # Use 2x the dataset length to ensure enough samples for training
        dataset_length = len(dataset_df) * 2
    else:
        dataset_length = len(dataset_df)

    for i in range(dataset_length):
        # Sample from dataset (with replacement for longer training)
        sample_idx = i % len(dataset_df)
        row = dataset_df.iloc[sample_idx]

        # Prepare sample data for MLP classification
        sample_data = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "keywords": row["keywords"]
        }

        # Get MLP prediction
        mlp_probability = mlp_classifier_inference(sample_data)

        # Determine label based on MLP probability
        predicted_label = 1 if mlp_probability > 0.5 else 0

        # Prepare sample with MLP-generated label
        sample = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "label": predicted_label,
            "classes": "no|yes",
            "class_confidences": f"{1.0-mlp_probability:.3f}|{mlp_probability:.3f}",
            "keywords": row["keywords"],
            "task": row["task"],
            "mlp_probability": mlp_probability
        }

        # Format messages for chat template
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        yield {
            "prompt": prompt,
            "label": sample["label"],
            "classes": sample["classes"],
            "class_confidences": sample["class_confidences"],
            "keywords": sample["keywords"],
            "task": sample["task"],
            "system_prompt": sample["system_prompt"],
            "user_prompt": sample["user_prompt"],
        }


# # Rewards
# `reward_answer_against_label` rewards the answer provided by the model (typically yes/no according to our prompts) by assigning the probability of the selected answer as estimated by the simplified VCM soft verifier.
#
# `composite_formatting_reward` makes sure formatting of the LLM output is compliant to our expectations and guidelines expressed in system prompt.
#
# `keywords_mentioned_in_think` makes sure specific keywords (typically gene names) are mentioned during reasoning.
#
# `compute_simple_reward` is used by the trainer to assign a reward to a generated trace.


def reward_answer_against_label(completion: str, classes: str, class_confidence: str) -> float:
    """Compute reward based on whether answer matches expected label"""
    answer = extract_binary_answer(completion)
    if answer is None:
        return 0.0

    answer = "yes" if answer else "no"
    possible_classes = classes.split("|")
    confidences = [float(c) for c in class_confidence.split("|")]

    for label, conf in zip(possible_classes, confidences):
        if answer == label.strip().lower():
            return conf

    return 0.0


def composite_formatting_reward(text: str, use_go: bool = False) -> float:
    """Compute composite formatting reward based on multiple checks"""
    at_least_one_think = has_at_least_one_think(text)
    has_tags = has_any_tag(text)

    checks = [
        at_least_one_think,
        low_untagged_ratio(text),
        is_not_too_long(text),
        has_one_answer(text),
        answer_after_thinks(text),
        thinks_have_text(text) * at_least_one_think,
        no_nested_tags(text) * has_tags,
        has_limited_thinks(text) * at_least_one_think,
        all_tags_properly_closed(text) * has_tags,
        ends_with_answer(text),
        starts_with_think(text),
    ]

    # Remove start_with_think dependency if using GO ontology
    if use_go:
        checks = checks[:-1]

    return sum(checks) / len(checks)

def keywords_mentioned_in_think(text: str, keywords: str) -> float:
    """Check how many keywords are mentioned in think sections"""
    keyword_list = [k for k in keywords.split("|") if k]

    if not keyword_list:
        return 1.0

    think_contents = extract_think(text)
    if not think_contents:
        return 0.0

    found_keywords = 0
    for keyword in keyword_list:
        if keyword in think_contents:
            found_keywords += 1

    return found_keywords / len(keyword_list)


def compute_simple_reward(
    completions: List[str],
    label: List[int],
    classes: List[str],
    class_confidences: List[str],
    keywords: List[str],
    **kwargs
) -> List[float]:
    """Compute rewards for model completions using format, mention, and answer rewards"""
    scores = []

    global STEP_COUNT

    for completion, lbl, class_list, confidences, keyword_list in zip(
        completions, label, classes, class_confidences, keywords
    ):
        # Format reward: checks proper tag structure
        format_reward = composite_formatting_reward(completion, use_go=False)

        # Mention reward: checks if keywords are mentioned in think sections
        mention_reward = keywords_mentioned_in_think(completion, keyword_list)

        # Answer reward: checks if answer matches expected label
        answer_reward = reward_answer_against_label(completion, class_list, confidences)

        # Combine rewards (answer reward gets 2x weight as it's most important)
        total_score = format_reward + 2.0 * answer_reward + mention_reward

        scores.append(total_score)

        # Debug prints every 100 steps to monitor model outputs
        if STEP_COUNT % 100 == 0:
            print("\n" + "="*80)
            print(f"DEBUG: Sample {STEP_COUNT} - Step {STEP_COUNT}")
            print("="*80)

            # Print the completion to see what the model generated
            print(f"MODEL OUTPUT:")
            print(f"{completion}")
            print()

            # Print reward breakdown
            print(f"REWARD BREAKDOWN:")
            print(f"  Format reward: {format_reward:.3f}")
            print(f"  Mention reward: {mention_reward:.3f}")
            print(f"  Answer reward: {answer_reward:.3f}")
            print(f"  Total score: {total_score:.3f}")
            print()

            # Print expected vs predicted
            print(f"EXPECTED vs PREDICTED:")
            print(f"  VCM binarized label: {lbl}")
            print(f"  Possible classes: {class_list}")
            print(f"  Label VCM confidences: {confidences}")
            print(f"  Keywords: {keyword_list}")
            print()

            # Print reward details
            print(f"REWARD DETAILS:")
            print(f"  Answer extraction: {extract_binary_answer(completion)}")
            print(f"  Think content: {extract_think(completion)[:100]}...")
            print("="*80 + "\n")
    STEP_COUNT += 1

    return scores


# # Training


print("Starting RBIO training with streaming MLP labeling...")

# Load and prepare dataset
print("Loading dataset...")
dataset_df = load_and_prepare_dataset(DATASET_PATHS)

# Load MLP classifier
print("Loading MLP classifier...")
load_mlp_classifier(MLP_MODEL_PATH, EMBEDDING_FILE, MLPClassifier)

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

# Create streaming dataset generator
print("Creating streaming dataset generator...")
dataset = Dataset.from_generator(
    create_mlp_labeled_dataset_generator,
    gen_kwargs={
        "dataset_df": dataset_df,
        "tokenizer": tokenizer,
        "balance_pos_neg": True,
    },
)

# Create training configuration
print("Setting up training configuration...")
training_config = create_training_config(
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_steps=N_STEPS,
    save_every=SAVE_EVERY
)

# Create trainer
print("Creating GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=compute_simple_reward,
    args=training_config,
    train_dataset=dataset,
)

# Start training
print(f"Starting training for {N_STEPS} steps...")
trainer.train()

print("Training completed!")


# ## Notes
# - This code is a different implementation compared to the code that has been used to train the methods discussed in our paper "Rbio: ...."
# - If you are interested only in using the perturbation data we employ in this dataset, please refer to the original repository https://github.com/genentech/PerturbQA and cite the work from our colleagues at Genentech accordingly
