import json
import re
import logging

logger = logging.getLogger(__name__)

def safe_extract_json(text: str):

    try:
        return extract_json(text)
    except Exception:
        logger.debug(
            "Failed to extract JSON from model output (first 200 chars): %s",
            text[:200],
        )
        return None
    
def extract_json(text: str) -> dict:
    """
    Robustly extract the first valid JSON object from model output.
    Handles markdown, extra text, and nested JSON.
    """

    if not text:
        raise ValueError("Empty model response")

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for i in range(start, len(text)):

        if text[i] == "{":
            depth += 1

        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)

                except Exception:
                    logger.debug(
                        "JSON candidate failed to parse: %s",
                        candidate[:200],
                    )
                    break

    raise ValueError(
        f"Failed to extract JSON from model output:\n{text[:500]}"
    )