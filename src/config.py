"""Experiment settings. Anything we may want to vary between runs lives here, not in code."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Config:
    experiment: str
    embedding_model: str


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    return Config(**yaml.safe_load(path.read_text(encoding="utf-8")))
