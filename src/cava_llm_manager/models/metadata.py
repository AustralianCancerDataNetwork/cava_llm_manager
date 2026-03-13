from pydantic import BaseModel, HttpUrl, Field, field_validator
from datetime import date
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class RegistryView:
    _models: Dict[str, "ModelMetadata"]
    _prompts: Dict[str, dict]
    _system_prompts: Dict[str, str]

    @property
    def models(self) -> list[str]:
        return sorted(self._models)

    @property
    def prompts(self) -> list[str]:
        return sorted(self._prompts)

    @property
    def system_prompts(self) -> list[str]:
        return sorted(self._system_prompts)

    @property
    def model_metadata(self) -> list["ModelMetadata"]:
        return list(self._models.values())

    def set_model_host_label(self, model_id: str, label: str) -> None:
        self._models[model_id].host_label = label

class ModelMetadata(BaseModel):
    name: str
    version: str

    provider: str
    architecture: str

    quantization: str | None = None
    context_window: int | None = None

    description: str | None = None
    link: HttpUrl | None = None

    date_added: date | None = None
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    parameter_count: int | None
    family: str | None

    host_label: str | None = None

    @property
    def server_label(self) -> str:
        return self.host_label or self.id

    @property
    def id(self) -> str:
        parts = [
            self.name.lower(),
            self.version.lower(),
        ]

        if self.quantization:
            parts.append(self.quantization.lower())

        return ":".join(parts)
        
    @field_validator("architecture")
    @classmethod
    def normalize_arch(cls, v):
        return v.lower()
    
    @field_validator("quantization")
    @classmethod
    def normalize_quant(cls, v):
        if v:
            return v.lower()
        return v
    
