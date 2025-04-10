from utils.verifiers import check_math_solution, test_code_solution, test_vcm_task
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


def perturb_reward_func(prompts, completions, task, **kwargs):
    """
    Perturbation reward function. Checks a number of completions in response to a list of perturbation prompts and assigns rewards 
    
    Args:
        prompts: prompts to the model
        completions: model completions to the prompts, each corresponding to a potential solution to a perturbation question
    Returns:
        rewards: list of rewards accumulated by checking the prompts
    """
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "perturbation":
            # invoke ML model
            works = test_vcm_task(prompt, completion, vcm, task)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards


def reward_len(completions, **kwargs):
    """
    Simple reward function. Checks a number of completions and assigns rewards if the completions are > 20 characters
    
    Args:
        completions: model completions to the prompts
    Returns:
        rewards: list of rewards accumulated by checking the prompts
    """
    rewards = [-abs(20 - len(completion)) for completion in completions]
    return rewards


def is_tag_usage_valid(text: str, tag: str):
    # this checks that if a tag is opened, it is also closed Eg. <think></think>
    # and that tags are never nested
    pattern = fr'</?{tag}>'
    tags = list(re.finditer(pattern, text))
    stack = []
    for t in tags:
        if t.group() == f'<{tag}>':
            if stack:
                return 0  # nested tag
            stack.append(t.start())
        else:  # closing tag
            if not stack:
                return 0  # unmatched closing tag
            stack.pop()
    return 1 if not stack else 0


def has_tag(text: str, tag: str):
    # this checks that tags are present
    pattern = fr'<{tag}>.*?</{tag}>'
    return 1 if re.search(pattern, text, re.DOTALL) else 0


def is_single_valid_answer(text: str):
    pattern = r'</?answer>'
    tags = list(re.finditer(pattern, text))

    if len(tags) != 2:
        return 0  # must have exactly one opening and one closing tag

    # Ensure correct order and no nesting
    return 1 if tags[0].group() == '<answer>' and tags[1].group() == '</answer>' else 0


def formatting_reward(completions: list, **kwargs):
    rewards = []
    for completion in completions:
        reward = (is_tag_usage_valid(completion, 'think') +
                  is_tag_usage_valid(completion, 'answer') +
                  has_tag(completion, 'think') +
                  has_tag(completion, 'answer') +
                  is_single_valid_answer(completion)
                  ) / 5.0

        rewards.append(reward)

    return rewards


def is_answer_yes(text: str):
    # this checks if the answer is positive
    match = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower() == 'yes'
    return False


def correctness_reward(completions, labels):
    rewards = []
    for completion, label in zip(completions, labels):
        rewards.append(float(is_answer_yes(completion) == bool(label)))

    return rewards