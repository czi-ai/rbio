from utils.verifiers import check_math_solution, test_code_solution, test_vcm_task

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