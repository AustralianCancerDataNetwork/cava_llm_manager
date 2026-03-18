from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMOutputModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    @staticmethod
    def coerce_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return [value]
        return []


class LLMReportModel(LLMOutputModel):
    report_id: int = Field(
        default=0,
        description="1-based position of the source item in the batch",
    )
    input_text: str = Field(
        default="",
        description="Exact input text sent to the LLM for this report",
    )

    @field_validator("report_id", mode="before")
    @classmethod
    def coerce_report_id(cls, value: Any) -> int:
        if value in (None, ""):
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
