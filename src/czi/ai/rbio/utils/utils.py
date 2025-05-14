import re


def extract_answer(text):
    found = re.search(r"<answer>\s*(yes|no)\s*</answer>", text, re.IGNORECASE)
    if found:
        if found.group(1).strip().lower() == "yes":
            return True
        if found.group(1).strip().lower() == "no":
            return False

    return None


def extract_think(text, separator="\n"):
    think_contents = re.findall(
        r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE
    )
    return separator.join(think_contents).strip()


def extract_gene_info(text, gene):
    # print(text)
    found = re.search(f'<gene_info>\s*(.*?{gene}.*?)\s*</gene_info>', text, re.IGNORECASE)
    if found:
        matches = found.group(1).strip()
        if gene in matches:
            return matches
    return f"No information."