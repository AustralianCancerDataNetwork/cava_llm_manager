from dataclasses import dataclass
from typing import Type, Optional
from pathlib import Path

from cava_llm_manager.models.registry import get_model
from cava_llm_manager.schemas.genomic.mutations import BatchResult
from pydantic import BaseModel

@dataclass
class PipelineSpec:

    name: str

    model_id: str
    system_prompt: str
    fewshot_id: str | None = None

    return_schema: Type[BaseModel] | None = None
    inject_schema: bool = False