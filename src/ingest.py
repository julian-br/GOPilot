"""The EBM catalogue as documents, and the vector index built from them."""

from langchain_core.documents import Document

from src.db.vectors import open_store
from src.ebm import load_gops


def build_index(embedding_model: str) -> int:
    docs = documents()
    store = open_store(embedding_model, force_recreate=True)
    try:
        store.add_documents(docs)
    finally:
        store.client.close()
    return len(docs)


def documents() -> list[Document]:
    return [
        Document(
            page_content=gop.embedding_text,
            metadata={"code": gop.code, "specialties": list(gop.specialties)},
        )
        for gop in load_gops()
    ]
