from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint
from ..utils.utils import extract_answer
import pd
from tqdm import tqdm
import os


def benchmark_grpo_trained(
        dataset_path: os.PathLike,
        model_name: str,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        answer = extract_answer(response)

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
    print(f'DONE WITH {model_name}')

    return stats