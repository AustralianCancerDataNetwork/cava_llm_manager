from pathlib import Path
import logging
from functools import wraps
import threading

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

_INITIALIZED = False
_INIT_LOCK = threading.Lock()

def ensure_initialized():

    global _INITIALIZED

    if _INITIALIZED:
        return

    with _INIT_LOCK:

        if _INITIALIZED:
            return

        logger.info("Initializing cava_llm_manager artifacts")

        from .models.loader import (
            load_models,
            load_prompts,
            load_system_prompts,
        )
        load_models(ARTIFACTS_DIR / "models")
        load_prompts(ARTIFACTS_DIR / "prompts" / "fewshot")
        load_system_prompts(ARTIFACTS_DIR / "prompts" / "system")

        _INITIALIZED = True

        logger.info("Artifact loading complete")

def requires_init(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):

        ensure_initialized()

        return fn(*args, **kwargs)

    return wrapper