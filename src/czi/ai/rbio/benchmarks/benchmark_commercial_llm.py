import os
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd
from tqdm import tqdm

from czi.ai.rbio.utils.utils import extract_answer


def benchmark_commercial_llm(
    dataset_path: os.PathLike,
    llm_model: str,
    llm_endpoint: str = os.environ["LLM_ENDPOINT_URL"],
    llm_api_key: str = os.environ["LLM_ENDPOINT_KEY"],
):

    llm = ChatOpenAI(
        base_url=llm_endpoint, api_key=llm_api_key, model=llm_model, temperature=0.0
    )

    dataset = pd.read_csv(dataset_path)

    # Initialize list to store results
    results: List[Dict[str, Any]] = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        system_prompt = row["system_prompt"]
        user_prompt = row["user_prompt"]
        label = row["label"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        while True:
            try:
                response = llm.invoke(messages)
                break
            except:
                time.sleep(5)

        answer = extract_answer(response)

        bool_label = label == 1

        # Record result
        result = {
            "prompt": f"{system_prompt}\n{user_prompt}",
            "completion": response,
            "answer": answer,
            "binary_answer": 1 if answer is True else (0 if answer is False else -1),
            "ground_truth": bool_label,
        }
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return None


@click.command()
@click.option("--dataset-path", help="Dataset CSV file path", required=True)
@click.option("--llm-model", help="The name of the LLM model", required=True)
@click.option("--llm-endpoint", help="URL of commercial model endpoint", required=True)
@click.option(
    "--llm-api-key",
    help="The API key to access the LLM via the endpoint",
    required=True,
)
@click.option("--output-path", help="Path to save results (CSV file)", required=True)
def benchmark(
    dataset_path: os.PathLike,
    llm_model: str,
    llm_endpoint: str,
    llm_api_key: str,
    output_path: os.PathLike,
):
    benchmark_commercial_llm(
        dataset_path=dataset_path,
        llm_model=llm_model,
        llm_endpoint=llm_endpoint,
        llm_api_key=llm_api_key,
        output_path=output_path,
    )


if __name__ == "__main__":
    benchmark()
