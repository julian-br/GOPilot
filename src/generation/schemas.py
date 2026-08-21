from dataclasses import dataclass

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    code: str = Field(description="Five-digit EBM GOP billing code")
    reason: str = Field(description="Brief reason for choosing this code")


class RecommendationResult(BaseModel):
    recommendations: list[Recommendation] = Field(
        description="Recommended EBM GOP billing codes"
    )


@dataclass(frozen=True)
class RecommendationRun:
    result: RecommendationResult
    reasoning: str | None
