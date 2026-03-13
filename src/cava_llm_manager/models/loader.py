import yaml # type: ignore
from pathlib import Path
import logging

from .registry import register_model, MODEL_REGISTRY, PROMPT_REGISTRY, SYSTEM_PROMPT_REGISTRY
from .metadata import ModelMetadata
from ..utils.prompt_utils import load_system_prompt

logger = logging.getLogger(__name__)

def load_models(model_dir: Path) -> None:
    logger.info("Loading model metadata from %s", model_dir)
    MODEL_REGISTRY.clear()
    files = sorted(model_dir.glob("*.yaml"))
    if not files:
        logger.warning("No model YAML files found in %s", model_dir)
        return 
    
    for file in files:
        logger.debug("Reading model config: %s", file.name)
        try:
            data = yaml.safe_load(file.read_text())
            model = ModelMetadata(**data)
            register_model(model)
            logger.info(
                "Registered model: id=%s provider=%s architecture=%s",
                model.id,
                model.provider,
                model.architecture,
            )
        except Exception as e:
            logger.error("Error loading model from %s: %s", file.name, str(e))
    logger.info("Loaded %d models into registry", len(MODEL_REGISTRY))

def load_prompts(prompt_dir: Path) -> None:
    PROMPT_REGISTRY.clear()
    logger.info("Loading prompts from %s", prompt_dir)
    for file in prompt_dir.glob("*.yaml"):
        logger.debug("Loading prompt: %s", file.stem)
        data = yaml.safe_load(file.read_text())
        PROMPT_REGISTRY[file.stem] = data
    logger.info("Loaded %d prompts", len(PROMPT_REGISTRY))

def load_system_prompts(system_prompt_dir: Path) -> None:
    SYSTEM_PROMPT_REGISTRY.clear()
    logger.info("Loading system prompts from %s", system_prompt_dir)
    for file in system_prompt_dir.glob("*.py"):
        logger.debug("Loading system prompt: %s", file.stem)
        data = load_system_prompt(file.stem)
        SYSTEM_PROMPT_REGISTRY[file.stem] = data
    logger.info("Loaded %d system prompts", len(SYSTEM_PROMPT_REGISTRY))