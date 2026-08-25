from src.config import Config, NoRagConfig, RetrievalConfig, WorkflowConfig
from src.db import open_store
from src.generation.client import open_chat_model
from src.generation.predictors.agent import AgentPredictor
from src.generation.predictors.base import Predictor
from src.generation.predictors.no_rag import NoRagPredictor
from src.generation.predictors.rag import RagPredictor
from src.generation.predictors.workflow import WorkflowPredictor
from src.retrieval import build_retriever


def build_predictor(config: Config) -> Predictor:
    model = open_chat_model(
        config.llm_provider,
        config.llm_model,
        config.llm_reasoning,
    )
    if isinstance(config, NoRagConfig):
        return NoRagPredictor(model, config.catalogue_quarter)

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
        if isinstance(config, WorkflowConfig):
            predictor = WorkflowPredictor(
                model,
                retriever,
                config.catalogue_quarter,
                config.max_services,
                config.max_candidates_per_path,
                store.client.close,
            )
        elif config.generation_strategy == "rag":
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
