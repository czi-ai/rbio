from typing import Callable, List

from openai import OpenAI
import os
import pandas as pd

# Create client with custom base URL
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url="https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints"
)

def benchmark_openai(csv_path: os.PathLike, metrics: List[Callable]):
    dataset = pd.read_csv('file_path.csv')

    for index, row in dataset.iterrows():
        system_prompt = row['system_prompt']
        user_prompt = row['user_prompt']
        label = row['label']

        response = client.chat.completions.create(
            model="gpt-4.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )