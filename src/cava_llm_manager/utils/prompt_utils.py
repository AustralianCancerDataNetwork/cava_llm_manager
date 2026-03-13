from typing import List, Dict
import yaml
from pathlib import Path
from importlib import import_module
from pydantic import BaseModel, schema
import json
import logging

logger = logging.getLogger(__name__)

def load_yaml_artifact(path: Path):
    logger.debug("Loading YAML artifact: %s", path)
    return yaml.safe_load(path.read_text())

def load_fewshot_examples(path: Path):
    data = load_yaml_artifact(path)
    return data.get("examples", [])

def get_root_collection_name(schema: type[BaseModel]) -> str:
    """
    Return the top-level collection field name of the schema.
    Example: GenomicBatchResult -> 'reports'
    """

    fields = list(schema.model_fields.keys())

    if len(fields) != 1:
        raise ValueError(
            f"{schema.__name__} must have exactly one top-level field"
        )

    return fields[0]

def build_prompt(
    items: List[str],
    schema: type[BaseModel],
    fewshot: List[Dict] | None = None
):

    root = get_root_collection_name(schema)
    label = root.capitalize()

    sections = []

    if fewshot:
        sections.append("Examples:\n")

        for example in fewshot:

            sections.append(f"{label}:")
            sections.append(f"1. {example['input'].strip()}")

            sections.append("Output (JSON):")
            sections.append(
                json.dumps(example["output"], indent=2)
            )

    sections.append(f"\nNow extract results from these {root}.\n")

    sections.append(f"{label}:")

    for i, item in enumerate(items):
        sections.append(f"{i+1}. {item}")

    return "\n\n".join(sections)

def load_system_prompt(name: str) -> str:

    logger.debug("Loading system prompt module: %s", name)
    module = import_module(
        f"cava_llm_manager.artifacts.prompts.system.{name}"
    )

    return module.SYSTEM_PROMPT


def schema_to_prompt(schema: type[BaseModel]) -> str:
    """
    Convert a Pydantic model to a JSON schema snippet suitable for prompting.
    """

    schema_dict = schema.model_json_schema()

    return (
        "Return JSON matching this schema exactly:\n\n"
        + json.dumps(schema_dict, indent=2)
    )