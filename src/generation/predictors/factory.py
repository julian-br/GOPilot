from typing import TYPE_CHECKING

from src.db import collection_quarter, open_store
from src.generation.client import open_chat_model
from src.generation.predictors.agent import AgentPredictor
from src.generation.predictors.base import Predictor
from src.generation.predictors.no_rag import NoRagPredictor
from src.generation.predictors.rag import RagPredictor
from src.retrieval import build_retriever

if TYPE_CHECKING:
    from src.config import Config


def build_predictor(config: "Config") -> Predictor:
    model = open_chat_model(config.llm_provider, config.llm_model)
    quarter = collection_quarter(config.ebm_collection)
    if config.generation_strategy == "no_rag":
        return NoRagPredictor(model, quarter)
    if config.generation_strategy in ("rag", "agent"):
        store = open_store(
            config.embedding_model,
            config.ebm_collection,
            config.retriever,
        )
        try:
            retriever = build_retriever(
                store,
                config.practice_specialty,
                config.top_k,
                config.reranker,
            )
            if config.generation_strategy == "rag":
                predictor = RagPredictor(
                    model,
                    retriever,
                    quarter,
                    store.client.close,
                )
            else:
                predictor = AgentPredictor(
                    model,
                    store,
                    retriever,
                    config.practice_specialty,
                    quarter,
                    store.client.close,
                )
        except Exception:
            store.client.close()
            raise
        return predictor
    raise ValueError(f"unknown generation strategy {config.generation_strategy!r}")
