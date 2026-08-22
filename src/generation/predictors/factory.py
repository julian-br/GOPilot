from src.config import Config, NoRagConfig, RetrievalConfig
from src.db import open_store
from src.generation.client import open_chat_model
from src.generation.predictors.agent import AgentPredictor
from src.generation.predictors.base import Predictor
from src.generation.predictors.no_rag import NoRagPredictor
from src.generation.predictors.rag import RagPredictor
from src.retrieval import build_retriever


def build_predictor(config: Config) -> Predictor:
    model = open_chat_model(config.llm_provider, config.llm_model)
    if isinstance(config, NoRagConfig):
        return NoRagPredictor(model, config.catalogue_quarter)

    if not isinstance(config, RetrievalConfig):
        raise TypeError(f"unsupported config type {type(config).__name__}")

    store = open_store(
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
                config.catalogue_quarter,
                store.client.close,
            )
        else:
            predictor = AgentPredictor(
                model,
                store,
                retriever,
                config.practice_specialty,
                config.catalogue_quarter,
                store.client.close,
            )
    except Exception:
        store.client.close()
        raise
    return predictor
