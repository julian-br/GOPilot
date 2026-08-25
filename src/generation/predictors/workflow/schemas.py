from pydantic import BaseModel, Field


class PerformedService(BaseModel):
    description: str = Field(
        description="General description of exactly one explicitly performed service",
    )


class PerformedServices(BaseModel):
    performed_services: list[PerformedService] = Field(
        max_length=12,
        description="Separate generally worded services explicitly performed in the dictation",
    )
