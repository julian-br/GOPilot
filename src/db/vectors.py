"""Vector store over the EBM catalogue."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models

from src.paths import FASTEMBED, QDRANT

if TYPE_CHECKING:
    from src.retrieval import RetrieverName

RETRIEVAL_MODES: dict[RetrieverName, RetrievalMode] = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}


def open_store(
    embedding_model: str, retriever: RetrieverName = "hybrid", force_recreate: bool = False
) -> QdrantVectorStore:
    """One collection per embedding model; their vectors have different dimensions."""
    QDRANT.mkdir(parents=True, exist_ok=True)
    FASTEMBED.mkdir(parents=True, exist_ok=True)
    embedding = OllamaEmbeddings(model=embedding_model)
    sparse_embedding = FastEmbedSparse(
        model_name="Qdrant/bm25",
        cache_dir=str(FASTEMBED),
        language="german",
    )
    store = QdrantVectorStore.construct_instance(
        embedding=embedding,
        sparse_embedding=sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID,
        client_options={"path": str(QDRANT)},
        collection_name=embedding_model.replace(":", "_"),
        vector_name="dense",
        sparse_vector_name="sparse",
        sparse_vector_params={
            "index": models.SparseIndexParams(on_disk=False),
            "modifier": models.Modifier.IDF,
        },
        force_recreate=force_recreate,
    )
    mode = RETRIEVAL_MODES[retriever]
    if mode == RetrievalMode.HYBRID:
        return store

    return QdrantVectorStore(
        client=store.client,
        collection_name=store.collection_name,
        embedding=store.embeddings,
        sparse_embedding=store.sparse_embeddings,
        retrieval_mode=mode,
        vector_name=store.vector_name,
        sparse_vector_name=store.sparse_vector_name,
        content_payload_key=store.content_payload_key,
        metadata_payload_key=store.metadata_payload_key,
        validate_collection_config=False,
    )


def billable_filter(specialty: str) -> models.Filter:
    """Codes without a specialty list are billable by anyone, so they must be included."""
    return models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.specialties",
                match=models.MatchValue(value=specialty),
            ),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="metadata.specialties")),
        ]
    )
