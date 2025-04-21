def check_math_solution(prompt, completion):
    """
    Verifier for math solutions. Checks if a completion corresponding to a math solution is correct.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a solution to a math equation
    Returns:
        True if the math solution corresponds to the given prompt
    """
    # TO-DO: complete implementation
    return True


def test_code_solution(prompt, completion):
    """
    Verifier for coding. Tests if completion corresponding to a coding task is correct.
    Runs the code and returns True if code runs.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a program
    Returns:
        True if the completion corresponding to the generated code runs
    """
    # TO-DO: complete implementation
    return True


def test_vcm_task(prompt, completion, vcm, task):
    """
    Verifier for VCM task. Tests if completion corresponding to a VCM task is correct.
    Runs the task using VCM and returns True if task is verifiable by the VCM model.

    Args:
        prompt: prompt to the model
        completion: model completion to the prompt, corresponding to a program
        vcm: vcm model to use
    Returns:
        True if the completion is verifiable by the vcm
    """
    # TO-DO: complete implementation
    return True
