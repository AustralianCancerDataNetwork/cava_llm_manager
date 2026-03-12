from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class GenomicTestResult(str, Enum):
    positive = "positive"
    negative = "negative"
    equivocal = "equivocal"
    result_pending = "result_pending"
    unknown = "unknown"

class PDL1Expression(str, Enum):
    high = "high"
    low = "low"
    negative = "negative"
    unknown = "unknown"


class GenomicTest(BaseModel):

    genomic_marker: str = Field(
        description="Name of the genomic marker tested (e.g. EGFR, KRAS, BRAF, ALK)"
    )

    test_result: GenomicTestResult = Field(
        description="Outcome of the genomic test"
    )

    variant: Optional[str] = Field(
        default=None,
        description="Specific variant if mentioned, e.g. V600E or exon 19 deletion"
    )

    pdl1_expression: Optional[PDL1Expression] = Field(
        default=None,
        description="PDL1 expression category (only for PDL1 tests)"
    )

    evidence: Optional[str] = Field(
        default=None,
        description="Text span supporting the extraction"
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Optional model confidence score"
    )


class ReportResult(BaseModel):

    report_id: int

    tests: List[GenomicTest] = Field(default_factory=list)

    report: Optional[str] = Field(
        default=None,
        description="Original report text"
    )


class BatchResult(BaseModel):

    reports: List[ReportResult] = Field(default_factory=list)