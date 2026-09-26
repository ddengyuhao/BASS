import re


def option_letters(num_options):
    return [chr(65 + i) for i in range(max(num_options, 1))]


def format_options(options):
    """Formats options as 'A. ...' lines, skipping letters already present."""
    if not isinstance(options, (list, tuple)):
        return str(options)
    lines = []
    for i, opt in enumerate(options):
        opt = str(opt).strip()
        letter = chr(65 + i)
        if re.match(rf"^\(?{letter}[\.\):]\s*", opt):
            lines.append(opt)
        else:
            lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def extract_answer_from_text(text, num_options=4, default=None):
    """
    Extracts an option letter from a model response.
    Matches e.g. "The answer is A", "(A)", "A.".
    """
    if not text:
        return default
    letters = "".join(option_letters(num_options))
    text = text.strip()
    patterns = [
        rf'(?:answer|option)\s*(?:is|:)\s*[\(]?([{letters}])[\)]?',
        rf'(?:^|\s)[\(]?([{letters}])[\)]?[\.\s]*$',
        rf'^[\(]?([{letters}])[\)]?[\.\s]',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return default
