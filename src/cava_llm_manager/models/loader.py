import yaml # type: ignore
from pathlib import Path

from .registry import register_model, MODEL_REGISTRY, PROMPT_REGISTRY, SYSTEM_PROMPT_REGISTRY
from .metadata import ModelMetadata
from ..utils.prompt_utils import load_system_prompt

def load_models(model_dir: Path):

    MODEL_REGISTRY.clear()

    for file in sorted(model_dir.glob("*.yaml")):

        data = yaml.safe_load(file.read_text())

        model = ModelMetadata(**data)

        register_model(model)

def load_prompts(prompt_dir: Path):

    for file in prompt_dir.glob("*.yaml"):

        data = yaml.safe_load(file.read_text())

        PROMPT_REGISTRY[file.stem] = data

def load_system_prompts(system_prompt_dir: Path):

    for file in system_prompt_dir.glob("*.py"):
        data = load_system_prompt(file.stem)
        SYSTEM_PROMPT_REGISTRY[file.stem] = data