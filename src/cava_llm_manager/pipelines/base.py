from dataclasses import dataclass
from typing import Type
from pydantic import BaseModel

@dataclass
class PipelineSpec:

    name: str

    model_id: str
    system_prompt: str
    fewshot_id: str | None = None

    return_schema: Type[BaseModel] | None = None
    inject_schema: bool = False