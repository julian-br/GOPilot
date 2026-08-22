from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from src.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "rag.yaml"
GenerationStrategy = Literal["no_rag", "rag", "agent"]
RetrieverName = Literal["dense", "sparse", "hybrid"]


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    candidate_k: PositiveInt


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    experiment: str
    embedding_model: str
    generation_strategy: GenerationStrategy
    llm_provider: str
    llm_model: str
    practice_specialty: str
    retriever: RetrieverName
    top_k: PositiveInt | None = None
    reranker: RerankerConfig | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        if self.generation_strategy == "rag":
            if self.top_k is None:
                raise ValueError("top_k is required for RAG generation")
            if self.reranker and self.reranker.candidate_k < self.top_k:
                raise ValueError("reranker candidate_k must be at least top_k")
        elif self.generation_strategy == "no_rag" and self.reranker:
            raise ValueError("reranker is not supported without RAG")
        return self


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config.model_validate(values)
