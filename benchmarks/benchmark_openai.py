from typing import Callable, List

from openai import OpenAI
import os
import pandas as pd
import re

# Create client with custom base URL
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url="https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints"
)


def extract_answer_if_present(text):
    # this code is duplicated on purpose
    found = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
    if found:
        if found.group(1).strip().lower() == 'yes':
          return True
        if found.group(1).strip().lower() == 'no':
          return False

    return None


def benchmark_openai(csv_path: os.PathLike, metrics: List[Callable]):
    dataset = pd.read_csv('file_path.csv')

    stats = {
        'fp': 0,
        'fn': 0,
        'tp': 0,
        'tn': 0,
        'unanswered': 0,
    }

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

        answer = extract_answer_if_present(response)

        if answer is not None:
            if answer == (label == 1):
                if answer:
                    stats['tp'] += 1
                else:
                    stats['tn'] += 1
            else:
                if not answer and (label == 1):
                    stats['fn'] += 1
                else:
                    stats['fp'] += 1
        else:
            stats['unanswered'] += 1

    print(f'STATS HAVE BEEN GENERATED FOR DATASET {csv_path}')
    print(stats)

    return stats