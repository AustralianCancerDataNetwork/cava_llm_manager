from pydantic import Field, field_validator
from typing import Any, List, Optional
from ..base import LLMOutputModel, LLMReportModel
from ..soft_enum import SoftEnum


class GenomicTestResult(SoftEnum):
    positive = "positive"
    wildtype = "wildtype"
    negative = "negative"
    equivocal = "equivocal"
    result_pending = "result_pending"
    unknown = "unknown"

    @classmethod
    def fallback(cls) -> str:
        return cls.unknown.value

    @classmethod
    def normalisations(cls) -> dict[str, str]:
        return {
            "wild type": "wildtype",
            "wild-type": "wildtype",
            "wt": "wildtype",
            "low likelihood negative": "negative",
            "not detected": "negative",
            "mutation detected": "positive",
            "send off": "result_pending",
            "pending": "result_pending",
            "unsatisfactory": "unknown",
            "inconclusive": "equivocal",
        }

class GenomicTest(LLMOutputModel):
    genomic_marker: str = Field(description="Name of the genomic marker tested (e.g. EGFR, KRAS, BRAF, ALK)", default="")
    test_result: GenomicTestResult = Field(description="Outcome of the genomic test", default=GenomicTestResult.unknown)
    variant: Optional[str] = Field(default=None, description="Specific variant if mentioned (e.g. V600E, exon 19 deletion)")
    enum_errors: List[str] = Field(default_factory=list)

    @field_validator("genomic_marker", mode="before")
    @classmethod
    def coerce_genomic_marker(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

class GenomicReportResult(LLMReportModel):
    tests: List[GenomicTest] = Field(default_factory=list, description="Genomic mutation test results mentioned in the report")

class GenomicBatchResult(LLMOutputModel):
    reports: List[GenomicReportResult] = Field(default_factory=list)
