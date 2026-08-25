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
        description=(
            "Only affirmative EBM GOP billing recommendations for the current visit. "
            "Never include rejected, irrelevant, or merely considered codes."
        )
    )
