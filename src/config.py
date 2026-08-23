from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from src.ebm.version import collection_quarter, parse_quarter
from src.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "rag.yaml"
RetrieverName = Literal["dense", "sparse", "hybrid"]


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    candidate_k: PositiveInt


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    experiment: str
    llm_provider: Literal["ollama", "openrouter"]
    llm_model: str


class NoRagConfig(BaseConfig):
    generation_strategy: Literal["no_rag"]
    catalogue_quarter: str

    @field_validator("catalogue_quarter")
    @classmethod
    def validate_catalogue_quarter(cls, value: str) -> str:
        return parse_quarter(value)


class RetrievalConfig(BaseConfig):
    generation_strategy: Literal["rag", "agent"]
    ebm_collection: str
    practice_specialty: str
    retriever: RetrieverName
    top_k: PositiveInt
    reranker: RerankerConfig | None = None

    @model_validator(mode="after")
    def validate_retrieval_config(self) -> Self:
        collection_quarter(self.ebm_collection)
        if self.reranker and self.reranker.candidate_k < self.top_k:
            raise ValueError("reranker candidate_k must be at least top_k")
        return self

    @property
    def catalogue_quarter(self) -> str:
        return collection_quarter(self.ebm_collection)


Config: TypeAlias = Annotated[
    NoRagConfig | RetrievalConfig,
    Field(discriminator="generation_strategy"),
]
CONFIG_ADAPTER = TypeAdapter(Config)


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CONFIG_ADAPTER.validate_python(values)
