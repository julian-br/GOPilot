"""Vector store over the EBM catalogue."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from src.paths import FASTEMBED, QDRANT

if TYPE_CHECKING:
    from src.config import RetrieverName

RETRIEVAL_MODES: dict[RetrieverName, RetrievalMode] = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}
COLLECTION_PATTERN = re.compile(r"^ebm_(\d{4})_q([1-4])(?:_|$)")


def open_store(
    embedding_model: str,
    collection_name: str,
    retriever: RetrieverName = "hybrid",
    force_recreate: bool = False,
) -> QdrantVectorStore:
    QDRANT.mkdir(parents=True, exist_ok=True)
    FASTEMBED.mkdir(parents=True, exist_ok=True)
    if not force_recreate:
        client = QdrantClient(path=str(QDRANT))
        try:
            if not client.collection_exists(collection_name):
                raise ValueError(f"Qdrant collection {collection_name!r} does not exist")
        finally:
            client.close()
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
        collection_name=collection_name,
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


def collection_quarter(collection_name: str) -> str:
    match = COLLECTION_PATTERN.match(collection_name)
    if match is None:
        raise ValueError(
            f"invalid EBM collection {collection_name!r}; expected ebm_<year>_q<1-4>"
        )
    year, quarter = match.groups()
    return f"{quarter}/{year}"


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


def find_gop(store: QdrantVectorStore, code: str) -> Document | None:
    points, _ = store.client.scroll(
        collection_name=store.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=f"{store.metadata_payload_key}.code",
                    match=models.MatchValue(value=code),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None

    payload = points[0].payload or {}
    return Document(
        page_content=payload[store.content_payload_key],
        metadata=payload[store.metadata_payload_key],
    )
