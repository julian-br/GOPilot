from typing import TYPE_CHECKING

from src.db import open_store
from src.generation.client import open_chat_model
from src.generation.predictors.base import Predictor
from src.generation.predictors.no_rag import NoRagPredictor
from src.generation.predictors.rag import RagPredictor
from src.retrieval import build_retriever

if TYPE_CHECKING:
    from src.config import Config


def build_predictor(config: "Config") -> Predictor:
    model = open_chat_model(config.llm_provider, config.llm_model)
    if config.generation_strategy == "no_rag":
        return NoRagPredictor(model)
    if config.generation_strategy == "rag":
        store = open_store(config.embedding_model, config.retriever)
        retriever = build_retriever(store, config.practice_specialty, config.top_k)
        return RagPredictor(model, retriever, store.client.close)
    if config.generation_strategy == "agent":
        raise NotImplementedError(
            f"generation strategy {config.generation_strategy!r} is not implemented"
        )
    raise ValueError(f"unknown generation strategy {config.generation_strategy!r}")
