"""Find candidate GOPs for a dictation."""

from typing import Literal, get_args

from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore

from src.db.vectors import billable_filter

RetrieverName = Literal["dense", "sparse", "hybrid"]
RETRIEVER_NAMES = get_args(RetrieverName)


def build_retriever(store: QdrantVectorStore, specialty: str, k: int) -> BaseRetriever:
    return store.as_retriever(search_kwargs={"k": k, "filter": billable_filter(specialty)})
