from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator


class Recommendation(BaseModel):
    code: str = Field(
        description="Exactly five digits of the selected EBM GOP, never a list position",
    )
    reason: str = Field(description="Brief reason for choosing this code")

    @field_validator("code")
    @classmethod
    def validate_code(cls, code: str) -> str:
        if len(code) != 5 or not code.isdigit():
            raise ValueError("code must be a five-digit EBM GOP")
        return code


class RecommendationResult(BaseModel):
    recommendations: list[Recommendation] = Field(
        description="Recommended EBM GOP billing codes"
    )


@dataclass(frozen=True)
class RecommendationRun:
    result: RecommendationResult
    reasoning: str | None

    @classmethod
    def from_output(cls, output: dict[str, Any]) -> Self:
        if output["parsing_error"] is not None:
            raise output["parsing_error"]
        reasoning = output["raw"].additional_kwargs.get("reasoning_content")
        return cls(result=output["parsed"], reasoning=reasoning)
