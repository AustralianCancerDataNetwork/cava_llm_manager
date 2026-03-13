from typing import Dict
from .metadata import ModelMetadata
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

def get_model(model_id: str) -> ModelMetadata:
    return MODEL_REGISTRY[model_id]
