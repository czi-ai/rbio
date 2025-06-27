import ast
import os
import re

from torch.nn.functional import softmax

from czi.ai.rbio.utils.utils import extract_answer, extract_think


def reward_tf_TFs_prediction_based_on_marker_genes(label, completion):
    answer = extract_answer(completion)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0
    return answer_reward


def reward_tf_gene_prediction_based_on_TFs(
    transcription_factor, gene_monitored, completion, label
):
    answer = extract_answer(completion)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0

    return answer_reward


def reward_tf_gene_pmi(
    gene_perturbed, gene_monitored, completion, gene_pmis, gene2idx, label
):
    answer = extract_answer(completion)

    if answer is None:
        return 0
    # pmi = gene_pmis[gene2idx[gene_perturbed], gene2idx[gene_monitored]]

    # reward = pmi * (answer == True) + (1 - pmi) * (answer == False)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0

    return answer_reward


def reward_tf_TFs_prediction_based_on_marker_genes(label, completion):
    answer = extract_answer(completion)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0
    return answer_reward


def reward_tf_gene_prediction_based_on_TFs(
    transcription_factor, gene_monitored, completion, label
):
    answer = extract_answer(completion)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0

    return answer_reward


def reward_tf_gene_pmi(
    gene_perturbed, gene_monitored, completion, gene_pmis, gene2idx, label
):
    answer = extract_answer(completion)

    if answer is None:
        return 0
    # pmi = gene_pmis[gene2idx[gene_perturbed], gene2idx[gene_monitored]]

    # reward = pmi * (answer == True) + (1 - pmi) * (answer == False)
    if answer is not None:
        answer_reward = float(answer == label)
    else:
        answer_reward = 0

    return answer_reward


def reward_answer_against_label(
    completion: str, classes: str, class_confidence: str
) -> float:
    answer = extract_answer(completion)
    if answer is None:
        return 0.0

    answer = answer.strip().lower()

    possible_classes = classes.split("|")
    confidences = [float(c) for c in class_confidence.split("|")]

    for label, conf in zip(possible_classes, confidences):
        if answer == label.strip().lower():
            return conf

    return 0.0


def has_at_least_one_think(text):
    return 1 if re.search(r"<think>.*?</think>", text, re.DOTALL) else 0


def low_untagged_ratio(text):
    text_no_tags = re.sub(r"</?(think|answer)>", "", text)
    total_words = len(re.findall(r"\b\w+\b", text_no_tags))

    tagged_words = 0
    for tag in re.findall(r"<(think|answer)>(.*?)</\1>", text, re.DOTALL):
        tagged_words += len(re.findall(r"\b\w+\b", tag[1]))
    ratio = tagged_words / total_words if total_words else 0

    return ratio


def starts_with_think(text):
    return 1 if re.match(r"^\s*<think>", text) else 0


def is_not_too_long(text):
    word_count = len(re.findall(r"\b\w+\b", text))
    return 1 if word_count <= 200 else 200 / word_count


def has_one_answer(text):
    matches = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
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


def thinks_have_text(text):
    return (
        1
        if all(
            re.search(r"\S", match)
            for match in re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        )
        else 0
    )


def keywords_mentioned_in_think(text: str, keywords: str) -> float:
    """
    Checks how many keywords from the pipe-separated string are mentioned in the think sections.
    Returns a score between 0 and 1 representing the ratio of found keywords.
    """
    # Split keywords and filter out empty strings
    keyword_list = [k for k in keywords.split("|") if k]

    # If no keywords to check, return 1.0
    if not keyword_list:
        return 1.0

    think_contents = extract_think(text)

    # If no think sections, return 0.0
    if not think_contents:
        return 0.0

    # Count how many keywords are found in any think section
    found_keywords = 0

    for keyword in keyword_list:
        if keyword in think_contents:
            found_keywords += 1

    # Return the ratio of found keywords to total keywords
    return found_keywords / len(keyword_list)


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
    tag_pattern = re.finditer(r"</?(think|answer)>", text)

    for tag in tag_pattern:
        tag_text = tag.group()
        tag_type = re.match(r"</?(think|answer)>", tag_text).group(1)

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
    return 1 if re.search(r"</?(think|answer)>", text) else 0


def composite_formatting_reward(text):
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
        starts_with_think(text),
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
