import re

def extract_answer(text):
    found = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
    if found:
        if found.group(1).strip().lower() == 'yes':
            return True
        if found.group(1).strip().lower() == 'no':
            return False

    return None
