from typing import Dict
from .metadata import ModelMetadata
from ..bootstrap import requires_init
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, ModelMetadata] = {}
PROMPT_REGISTRY = {}
SYSTEM_PROMPT_REGISTRY = {}

def register_model(model: ModelMetadata) -> None:

    if model.id in MODEL_REGISTRY:
        logger.warning("Model with id %s already registered", model.id)
        return

    MODEL_REGISTRY[model.id] = model
    logger.debug("Model registered: %s", model.id)

@requires_init
def get_model(model_id: str) -> ModelMetadata:
    return MODEL_REGISTRY[model_id]

@requires_init
def get_model_registry() -> Dict[str, ModelMetadata]:
    return MODEL_REGISTRY

@requires_init
def get_prompt(prompt_id: str):
    return PROMPT_REGISTRY[prompt_id]

@requires_init
def get_system_prompt(system_prompt_id: str):
    return SYSTEM_PROMPT_REGISTRY[system_prompt_id]