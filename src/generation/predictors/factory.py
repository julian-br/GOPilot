from typing import TYPE_CHECKING

from src.generation.client import open_chat_model
from src.generation.predictors.base import Predictor
from src.generation.predictors.no_rag import NoRagPredictor

if TYPE_CHECKING:
    from src.config import Config


def build_predictor(config: "Config") -> Predictor:
    if config.generation_strategy == "no_rag":
        return NoRagPredictor(open_chat_model(config.llm_provider, config.llm_model))
    if config.generation_strategy in {"rag", "agent"}:
        raise NotImplementedError(
            f"generation strategy {config.generation_strategy!r} is not implemented"
        )
    raise ValueError(f"unknown generation strategy {config.generation_strategy!r}")
