from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml

from src.paths import ROOT
from src.retrieval import RETRIEVER_NAMES, RetrieverName

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
GenerationStrategy = Literal["no_rag", "rag", "agent"]
GENERATION_STRATEGIES = get_args(GenerationStrategy)


@dataclass(frozen=True)
class Config:
    experiment: str
    embedding_model: str
    generation_strategy: GenerationStrategy
    llm_provider: str
    llm_model: str
    practice_specialty: str
    retriever: RetrieverName
    top_k: int

    def __post_init__(self) -> None:
        if self.retriever not in RETRIEVER_NAMES:
            allowed = ", ".join(RETRIEVER_NAMES)
            raise ValueError(
                f"unknown retriever {self.retriever!r}; expected one of: {allowed}"
            )
        if self.generation_strategy not in GENERATION_STRATEGIES:
            allowed = ", ".join(GENERATION_STRATEGIES)
            raise ValueError(
                f"unknown generation strategy {self.generation_strategy!r}; "
                f"expected one of: {allowed}"
            )


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    return Config(**yaml.safe_load(path.read_text(encoding="utf-8")))
