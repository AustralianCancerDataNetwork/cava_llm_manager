from typing import Any

from pydantic import Field, field_validator

from .base import LLMOutputModel, LLMReportModel
from .soft_enum import SoftEnum


class ConfidenceDecision(SoftEnum):
    accept = "accept"
    revise = "revise"
    uncertain = "uncertain"

    @classmethod
    def fallback(cls) -> str:
        return cls.uncertain.value


class ConfidenceBand(SoftEnum):
    high = "high"
    medium = "medium"
    low = "low"
    unknown = "unknown"

    @classmethod
    def fallback(cls) -> str:
        return cls.unknown.value


class ConfidenceReview(LLMReportModel):
    decision: ConfidenceDecision = Field(default=ConfidenceDecision.uncertain)
    confidence: ConfidenceBand = Field(default=ConfidenceBand.unknown)
    issues: list[str] = Field(default_factory=list)
    rationale: str | None = None
    corrected_output: dict[str, Any] | None = None

    @field_validator("issues", mode="before")
    @classmethod
    def coerce_issues(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)


class ConfidenceReviewBatchResult(LLMOutputModel):
    reviews: list[ConfidenceReview] = Field(default_factory=list)

    @field_validator("reviews", mode="before")
    @classmethod
    def coerce_reviews(cls, value: Any) -> list[Any]:
        return cls.coerce_list(value)
