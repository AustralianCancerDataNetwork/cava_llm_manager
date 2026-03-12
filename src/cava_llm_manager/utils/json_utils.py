import json
import re


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from a model response.
    """

    m = re.search(r"\{.*\}", text, re.S)

    if not m:
        raise ValueError("No JSON found")

    return json.loads(m.group())

def safe_extract_json(text: str):

    try:
        return extract_json(text)
    except Exception:
        return None