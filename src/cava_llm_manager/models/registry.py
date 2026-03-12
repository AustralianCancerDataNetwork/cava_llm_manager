from typing import Dict
from .metadata import ModelMetadata

MODEL_REGISTRY: Dict[str, ModelMetadata] = {}
PROMPT_REGISTRY = {}
SYSTEM_PROMPT_REGISTRY = {}

def register_model(model: ModelMetadata):

    if model.id in MODEL_REGISTRY:
        raise ValueError(f"Duplicate model id: {model.id}")

    MODEL_REGISTRY[model.id] = model


def get_model(model_id: str) -> ModelMetadata:
    return MODEL_REGISTRY[model_id]
