from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class PDL1Expression(str, Enum):
    high = "high"
    low = "low"
    negative = "negative"
    unknown = "unknown"

class PDL1Test(BaseModel):
    expression: PDL1Expression = Field(
        description="PD-L1 expression category"
    )
    percent: Optional[int] = Field(
        default=None,
        description="PD-L1 tumour proportion score percentage if mentioned"
    )

class PDL1ReportResult(BaseModel):
    report_id: int
    pdl1_tests: List[PDL1Test] = Field(default_factory=list)


class PDL1BatchResult(BaseModel):
    reports: List[PDL1ReportResult] = Field(default_factory=list)