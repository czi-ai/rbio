import re

from torch.nn.functional import softmax

from czi.ai.rbio.model.verifiers import call_vcm, verify_gene_info, verify_gene_info_rouge_scores, verify_gene_info_llh
from czi.ai.rbio.utils.utils import extract_answer, extract_think, extract_gene_info


def reward_gene_similarity_via_vcm(
    gene_perturbed,
    gene_monitored,
    completion,
    task,
    gene2ensembl_id,
    vcm_model,
    gene_vocab,
):
    answer = extract_answer(completion)

    if answer is None:
        return 0

    p_works_vcm = (
        call_vcm(
            gene_perturbed, gene_monitored, gene2ensembl_id, vcm_model, gene_vocab, task
        )
        .detach()
        .numpy()
    )

    reward = (1.0 * (answer == True) * p_works_vcm) + (
        -1.0 * (answer == False) * p_works_vcm
    )

    return reward

def reward_gene_information_go_ontology_mention(
    gene,
    completion, 
    gene2annotations
):
    gene_info_llm = extract_gene_info(completion, gene)
    if gene_info_llm == 'No information.':
        reward = 0
    else:
        reward = verify_gene_info(gene_info_llm, gene, gene2annotations)
    return reward

def reward_gene_information_go_ontology_rouge_score(
    gene,
    completion, 
    gene2annotations,
    scorer
):
    gene_info_llm = extract_gene_info(completion, gene)
    if gene_info_llm == 'No information.':
        reward1, reward2, reward3 = 0, 0, 0
    else:
        reward1, reward2, reward3 = verify_gene_info_rouge_scores(gene_info_llm, gene, gene2annotations, scorer)
    return reward1, reward2, reward3


def reward_go_info_llh(
    gene,
    gene2annotations,
    model, 
    tokenizer,
    go_ontology_type
):
    llh = verify_gene_info_llh(gene, gene2annotations, model, tokenizer, go_ontology_type)
    return llh


def reward_answer_against_label(completion: str, label: bool):
    answer = extract_answer(completion)

    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0

    return answer_reward


def has_at_least_one_think(text):
    return 1 if re.search(r"<think>.*?</think>", text, re.DOTALL) else 0

def has_at_least_one_gene_info(text):
    return 1 if re.search(r"<gene_info>.*?</gene_info>", text, re.DOTALL) else 0


def low_untagged_ratio(text):
    text_no_tags = re.sub(r"</?(think|answer|gene_info)>", "", text)
    total_words = len(re.findall(r"\b\w+\b", text_no_tags))

    tagged_words = 0
    for tag in re.findall(r"<(think|answer|gene_info)>(.*?)</\1>", text, re.DOTALL):
        tagged_words += len(re.findall(r"\b\w+\b", tag[1]))
    ratio = tagged_words / total_words if total_words else 0

    return ratio


def starts_with_think(text):
    return 1 if re.match(r"^\s*<think>", text) else 0

def starts_with_gene_info(text):
    return 1 if re.match(r"^\s*<gene_info>", text) else 0


def is_not_too_long(text):
    word_count = len(re.findall(r"\b\w+\b", text))
    return 1 if word_count <= 200 else 200 / word_count


def has_one_answer(text):
    matches = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
    return 1 if len(matches) == 1 else 0

def has_cellular_component(text):
    matches = re.findall(r"<gene_info>.*?cellular.*?(component|location|localization).*?</gene_info>", text, re.DOTALL)
    return 1 if len(matches) == 1 else 0

def has_localizes_mention(text):
    matches = re.findall(r"<gene_info>.*?(localizes|location|cellular compartment|cellular structure|cellular entity|virion component|macromolecular complex).*?</gene_info>", text, re.DOTALL)
    return 1 if len(matches) == 1 else 0


def answer_after_thinks(text):
    think_tags = list(re.finditer(r"</think>", text))
    answer_match = re.search(r"<answer>", text)
    if not answer_match:
        return 0
    if not think_tags:
        return 0
    last_think_end = think_tags[-1].end()
    return 1 if answer_match.start() > last_think_end else 0

def answer_after_gene_info(text):
    think_tags = list(re.finditer(r"</gene_info>", text))
    answer_match = re.search(r"<answer>", text)
    if not answer_match:
        return 0
    if not think_tags:
        return 0
    last_think_end = think_tags[-1].end()
    return 1 if answer_match.start() > last_think_end else 0

def think_after_gene_info(text):
    gene_info_tags = list(re.finditer(r"</gene_info>", text))
    think_match = re.search(r"<think>", text)
    if not think_match:
        return 0
    if not gene_info_tags:
        return 0
    last_gene_info_end = gene_info_tags[-1].end()
    return 1 if think_match.start() > last_gene_info_end else 0

def gene_info_inside_think(text):
    think_tags = list(re.finditer(r"</think>", text))
    gene_info_match = re.search(r"</gene_info>", text)
    if not gene_info_match:
        return 0
    if not think_tags:
        return 0
    last_think_end = think_tags[-1].end()
    return 1 if gene_info_match.start() < last_think_end else 0


def thinks_have_text(text):
    return (
        1
        if all(
            re.search(r"\S", match)
            for match in re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        )
        else 0
    )

def gene_infos_have_text(text):
    return (
        1
        if all(
            re.search(r"\S", match)
            for match in re.findall(r"<gene_info>(.*?)</gene_info>", text, re.DOTALL)
        )
        else 0
    )


def genes_mentioned_in_think(text, gene_perturbed, gene_monitored):
    think_contents = re.findall(
        r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
    )

    for content in think_contents:
        score = int(gene_perturbed in content) + int(gene_monitored in content)
        if score > 0:
            return score / 2.0  # 0.5 or 1.0
    return 0.0


def no_nested_tags(text):
    # Match all <think>...</think> and <answer>...</answer> blocks
    blocks = re.finditer(r"<(think|answer)>(.*?)</\1>", text, re.DOTALL)

    for block in blocks:
        tag_type = block.group(1)
        inner_text = block.group(2)
        # Look for any nested <think> or <answer> inside the block
        if re.search(r"</?(think|answer)>", inner_text, re.DOTALL):
            return 0
    return 1


def all_tags_properly_closed(text):
    tag_stack = []
    tag_pattern = re.finditer(r"</?(think|answer|gene_info)>", text)

    for tag in tag_pattern:
        tag_text = tag.group()
        tag_type = re.match(r"</?(think|answer|gene_info)>", tag_text).group(1)

        if tag_text.startswith("</"):
            # closing tag
            if not tag_stack or tag_stack[-1] != tag_type:
                return 0  # orphan or mismatched closing tag
            tag_stack.pop()
        else:
            # opening tag
            tag_stack.append(tag_type)

    return 1 if not tag_stack else 0  # stack must be empty if all matched


def has_limited_thinks(text):
    matches = re.findall(r"<think>.*?</think>", text, re.DOTALL)
    return 1 if len(matches) <= 1 else 1 / (len(matches) * 4)


def ends_with_answer(text):
    return 1 if text.endswith("</answer>") else 0


def has_any_tag(text):
    return 1 if re.search(r"</?(think|answer|gene_info)>", text) else 0


def composite_formatting_reward(text):
    at_least_one_think = has_at_least_one_think(text)
    at_least_one_gene_info = has_at_least_one_gene_info(text)
    has_tags = has_any_tag(text)
    checks = [
        at_least_one_think,
        at_least_one_gene_info,
        low_untagged_ratio(text),
        is_not_too_long(text),
        has_one_answer(text),
        answer_after_thinks(text),
        answer_after_gene_info(text),
        think_after_gene_info(text),
        gene_infos_have_text(text),
        thinks_have_text(text) * at_least_one_think,
        no_nested_tags(text) * has_tags,
        has_limited_thinks(text) * at_least_one_think,
        starts_with_gene_info(text),
        all_tags_properly_closed(text) * has_tags,
        ends_with_answer(text),
    ]
    return sum(checks) / len(checks)  # normalized score from 0 to 1


def reasoning_advantage_reward(
    model, tokenizer, system_prompt, user_prompt, completion, label
):
    answer = extract_answer(completion)

    if answer is not None:
        if answer:
            answer = ["yes", " yes", "yes ", " yes "]
        else:
            answer = ["no", " no", "no ", " no "]

    else:
        return 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_with_tt = (
        prompt + " <think> " + extract_think(completion) + " </think> <answer>"
    )

    prompt_without_tt = prompt + " <think> </think> <answer>"

    def compute_score(prompt, token_ids):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )

        scores = []
        for token_id in token_ids:

            probability = softmax(outputs.logits[0][0])[token_id]

            scores.append(probability)

        return max(scores)

    t_ids = tokenizer(answer)["input_ids"]

    token_ids = []
    for t_id in t_ids:
        token_ids.extend(t_id)

    score_without_tt = compute_score(prompt_without_tt, token_ids=token_ids)

    score_with_tt = compute_score(prompt_with_tt, token_ids=token_ids)

    reasoning_advantage = score_with_tt - score_without_tt

    if reasoning_advantage > 0:
        reasoning_advantage = 1.0

    return reasoning_advantage
