from utils.verifiers import call_vcm
import re


def math_reward_func(prompts, completions, task, **kwargs):
    """
    Math reward function. Checks a number of completions in response to a list of math solutions and assigns rewards 
    
    Args:
        prompts: prompts to the model
        completions: model completions to the prompts, each corresponding to a potential solution to a math equation
    Returns:
        rewards: list of rewards accumulated by checking the prompts
    """
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards


def coding_reward_func(prompts, completions, task, **kwargs):
    """
    Coding reward function. Checks a number of completions in response to a list of coding prompts and assigns rewards 
    
    Args:
        prompts: prompts to the model
        completions: model completions to the prompts, each corresponding to a potential solution to a coding prompt
    Returns:
        rewards: list of rewards accumulated by checking the prompts
    """
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards


def vcm_reward_func(gene_perturbed, gene_monitored, completion, task, gene2ensembl_id, model, gene_vocab):
    """
    Perturbation reward function. Checks a number of completions in response to a list of perturbation prompts and assigns rewards 
    
    Args:
        prompts: prompts to the model
        completions: model completions to the prompts, each corresponding to a potential solution to a perturbation question
    Returns:
        rewards: list of rewards accumulated by checking the prompts
    """
    if task == "gene_similarity":
        # invoke ML model
        answer_matches = re.findall(r'<answer>.*?</answer>', completion, re.DOTALL)
        answer = [x.split('<answer>')[1].split('</answer>')[0].strip() for x in answer_matches]
        if len(answer) == 0:
            return -2.0 #functional mismatch
        p_works_vcm = call_vcm(gene_perturbed, gene_monitored, gene2ensembl_id, model, gene_vocab, task).detach().numpy()
        reward = (-1.0 * (answer[0] == 'yes') * p_works_vcm) + (1.0 * (answer[0] == 'no') * p_works_vcm)
    else:
        reward = 0.0
    return reward


def has_at_least_one_think(text):
    return 1 if re.search(r'<think>.*?</think>', text, re.DOTALL) else 0


def low_untagged_ratio(text):
    text_no_tags = re.sub(r'</?(think|answer)>', '', text)
    total_words = len(re.findall(r'\b\w+\b', text_no_tags))

    tagged_words = 0
    for tag in re.findall(r'<(think|answer)>(.*?)</\1>', text, re.DOTALL):
        tagged_words += len(re.findall(r'\b\w+\b', tag[1]))
    ratio = tagged_words / total_words if total_words else 0

    return ratio


def starts_with_think(text):
    return 1 if re.match(r'^\s*<think>', text) else 0


def is_not_too_long(text):
    word_count = len(re.findall(r'\b\w+\b', text))
    return 1 if word_count <= 200 else 200 / word_count


def has_one_answer(text):
    matches = re.findall(r'<answer>.*?</answer>', text, re.DOTALL)
    return 1 if len(matches) == 1 else 0


def answer_after_thinks(text):
    think_tags = list(re.finditer(r'</think>', text))
    answer_match = re.search(r'<answer>', text)
    if not answer_match:
        return 0
    if not think_tags:
        return 0
    last_think_end = think_tags[-1].end()
    return 1 if answer_match.start() > last_think_end else 0


def thinks_have_text(text):
    return 1 if all(re.search(r'\S', match) for match in re.findall(r'<think>(.*?)</think>', text, re.DOTALL)) else 0


def genes_mentioned_in_think(text, gene_perturbed, gene_monitored):
    think_contents = re.findall(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)

    for content in think_contents:
        score = int(gene_perturbed in content) + int(gene_monitored in content)
        if score > 0:
            return score / 2.0  # 0.5 or 1.0
    return 0.0


def no_nested_tags(text):
    # Match all <think>...</think> and <answer>...</answer> blocks
    blocks = re.finditer(r'<(think|answer)>(.*?)</\1>', text, re.DOTALL)

    for block in blocks:
        tag_type = block.group(1)
        inner_text = block.group(2)
        # Look for any nested <think> or <answer> inside the block
        if re.search(r'</?(think|answer)>', inner_text, re.DOTALL):
            return 0
    return 1


def all_tags_properly_closed(text):
    tag_stack = []
    tag_pattern = re.finditer(r'</?(think|answer)>', text)

    for tag in tag_pattern:
        tag_text = tag.group()
        tag_type = re.match(r'</?(think|answer)>', tag_text).group(1)

        if tag_text.startswith('</'):
            # closing tag
            if not tag_stack or tag_stack[-1] != tag_type:
                return 0  # orphan or mismatched closing tag
            tag_stack.pop()
        else:
            # opening tag
            tag_stack.append(tag_type)

    return 1 if not tag_stack else 0  # stack must be empty if all matched


def has_limited_thinks(text):
    matches = re.findall(r'<think>.*?</think>', text, re.DOTALL)
    return 1 if len(matches) <= 1 else 1 / (len(matches) * 4)


def ends_with_answer(text):
    return 1 if text.endswith('</answer>') else 0


def has_any_tag(text):
    return 1 if re.search(r'</?(think|answer)>', text) else 0


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
        ends_with_answer(text)
    ]
    return sum(checks) / len(checks)  # normalized score from 0 to 1