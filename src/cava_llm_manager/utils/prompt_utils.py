from typing import List, Dict
import yaml
from pathlib import Path
from importlib import import_module
from pydantic import BaseModel
import json

def load_yaml_artifact(path: Path):
    return yaml.safe_load(path.read_text())

def load_fewshot_examples(path: Path):
    data = load_yaml_artifact(path)
    return data.get("examples", [])

def build_prompt(reports: List[str], fewshot: List[Dict] | None = None):

    sections = []

    if fewshot:
        sections.append("Examples:\n")

        for example in fewshot:
            sections.append(f"Input:\n{example['input']}")
            sections.append(f"Output:\n{example['output']}")

    sections.append("Reports:")

    for i, r in enumerate(reports):
        sections.append(f"{i+1}. {r}")

    return "\n\n".join(sections)

def load_system_prompt(name: str):

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