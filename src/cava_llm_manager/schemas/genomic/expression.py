from pydantic import Field, field_validator
from typing import Any, List, Optional
from ..base import LLMOutputModel, LLMReportModel
from ..soft_enum import SoftEnum


class PDL1Expression(SoftEnum):
    high = "high"
    low = "low"
    negative = "negative"
    unknown = "unknown"

    @classmethod
    def fallback(cls) -> str:
        return cls.unknown.value

    @classmethod
    def normalisations(cls) -> dict[str, str]:
        return {
            "positive": "high",
            "strong": "high",
            "weak": "low",
            "absent": "negative",
            "0": "negative",
        }


class PDL1Test(LLMOutputModel):
    expression: PDL1Expression = Field(
        description="PD-L1 expression category",
        default=PDL1Expression.unknown,
    )
    percent: Optional[int] = Field(
        default=None,
        description="PD-L1 tumour proportion score percentage if mentioned"
    )

    @field_validator("expression", mode="before")
    @classmethod
    def parse_expression(cls, v, info):
        value, error = PDL1Expression.parse(v)
        if error:
            info.data.setdefault("enum_errors", []).append(error)
        return value

    enum_errors: List[str] = Field(default_factory=list)


class PDL1ReportResult(LLMReportModel):
    pdl1_tests: List[PDL1Test] = Field(default_factory=list)

    @field_validator("pdl1_tests", mode="before")
    @classmethod
    def coerce_pdl1_tests(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)


class PDL1BatchResult(LLMOutputModel):
    reports: List[PDL1ReportResult] = Field(default_factory=list)

    @field_validator("reports", mode="before")
    @classmethod
    def coerce_reports(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)
