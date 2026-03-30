from pydantic import Field
from typing import List, Optional
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
    enum_errors: List[str] = Field(default_factory=list)


class PDL1ReportResult(LLMReportModel):
    pdl1_tests: List[PDL1Test] = Field(default_factory=list)


class PDL1BatchResult(LLMOutputModel):
    reports: List[PDL1ReportResult] = Field(default_factory=list)
