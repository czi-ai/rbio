from tqdm import tqdm
from openai import OpenAI
from langchain_openai import ChatOpenAI
from ..utils.utils import extract_answer
import os
import pandas as pd
import time

# Create client with custom base URL
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url="https://czi-virtual-cells-dev-databricks-workspace.cloud.databricks.com/serving-endpoints"
)


def benchmark_commercial_llm(
        dataset_path: os.PathLike,
        llm_model: str,
        llm_endpoint:str = os.environ['LLM_ENDPOINT_URL'],
        llm_api_key:str = os.environ['LLM_ENDPOINT_KEY'],
):

    llm = ChatOpenAI(
        base_url=llm_endpoint,
        api_key=llm_api_key,
        model=llm_model,
        temperature=0.0
    )

    dataset = pd.read_csv(dataset_path)

    stats = {
        'fp': 0,
        'fn': 0,
        'tp': 0,
        'tn': 0,
        'unanswered': 0,
    }

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        system_prompt = row['system_prompt']
        user_prompt = row['user_prompt']
        label = row['label']

        messages = [
           ("system", system_prompt),
           ("human",  user_prompt)
        ]

        while True:
            try:
                ai_msg = llm.invoke(messages)
                break
            except:
                time.sleep(5)

        answer = extract_answer(ai_msg)

        bool_label = (label == 1)

        if answer is not None:
            if answer == True and bool_label == True:
                stats['tp'] += 1
            elif answer == True and bool_label == False:
                stats['fp'] += 1
            elif answer == False and bool_label == False:
                stats['tn'] += 1
            elif answer == False and bool_label == True:
                stats['fn'] += 1
        else:
            stats['unanswered'] += 1

    print(f'STATS HAVE BEEN GENERATED FOR DATASET {dataset_path}')
    print(stats)
    print(f'DONE WITH {llm_model}')

    return stats