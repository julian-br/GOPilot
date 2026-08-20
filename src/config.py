from dataclasses import dataclass
from pathlib import Path

import yaml

from src.paths import ROOT
from src.retrieval import RETRIEVER_NAMES, RetrieverName

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Config:
    experiment: str
    embedding_model: str
    practice_specialty: str
    retriever: RetrieverName
    top_k: int

    def __post_init__(self) -> None:
        if self.retriever not in RETRIEVER_NAMES:
            allowed = ", ".join(RETRIEVER_NAMES)
            raise ValueError(f"unknown retriever {self.retriever!r}; expected one of: {allowed}")


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    return Config(**yaml.safe_load(path.read_text(encoding="utf-8")))
