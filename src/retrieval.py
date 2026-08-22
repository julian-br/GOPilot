"""Find candidate GOPs for a dictation."""

from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

from src.config import RerankerConfig
from src.db.vectors import billable_filter

class LocalCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return self.model.predict(text_pairs).tolist()


def build_retriever(
    store: QdrantVectorStore,
    specialty: str,
    k: int,
    reranker_config: RerankerConfig | None,
) -> BaseRetriever:
    retriever = store.as_retriever(
        search_kwargs={
            "k": reranker_config.candidate_k if reranker_config else k,
            "filter": billable_filter(specialty),
        }
    )
    if reranker_config is None:
        return retriever

    from langchain_classic.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    reranker = CrossEncoderReranker(
        model=LocalCrossEncoder(reranker_config.model),
        top_n=k,
    )
    return ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=reranker,
    )
