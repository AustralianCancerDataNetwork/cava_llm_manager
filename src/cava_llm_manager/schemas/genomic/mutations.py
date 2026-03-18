from pydantic import Field, field_validator
from typing import Any, List, Optional
from ..base import LLMOutputModel, LLMReportModel
from ..soft_enum import SoftEnum

# class GenomicTestResult(str, Enum):
#     positive = "positive"
#     wildtype = "wildtype"
#     negative = "negative"
#     equivocal = "equivocal"
#     result_pending = "result_pending"
#     unknown = "unknown"


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

    @field_validator("test_result", mode="before")
    @classmethod
    def parse_enum(cls, v: Any, info):
        value, error = GenomicTestResult.parse(v)
        if error:
            info.data.setdefault("enum_errors", []).append(error)
        return value

class GenomicReportResult(LLMReportModel):
    tests: List[GenomicTest] = Field(default_factory=list, description="Genomic mutation test results mentioned in the report")

    @field_validator("tests", mode="before")
    @classmethod
    def coerce_tests(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)

class GenomicBatchResult(LLMOutputModel):
    reports: List[GenomicReportResult] = Field(default_factory=list)

    @field_validator("reports", mode="before")
    @classmethod
    def coerce_reports(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)
