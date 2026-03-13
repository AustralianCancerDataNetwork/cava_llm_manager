from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class GenomicTestResult(str, Enum):
    positive = "positive"
    negative = "negative"
    equivocal = "equivocal"
    result_pending = "result_pending"
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
        description="Specific variant if mentioned (e.g. V600E, exon 19 deletion)"
    )

class GenomicReportResult(BaseModel):

    report_id: int

    tests: List[GenomicTest] = Field(
        default_factory=list,
        description="Genomic mutation test results mentioned in the report"
    )

class GenomicBatchResult(BaseModel):

    reports: List[GenomicReportResult] = Field(default_factory=list)