"""The EBM catalogue as documents, and the vector index built from them."""

from langchain_core.documents import Document

from src.db.vectors import collection_quarter, open_store
from src.ebm import load_gops, load_quarter


def build_index(collection_name: str) -> int:
    """Replace a validated EBM collection with the currently downloaded catalogue."""
    quarter = collection_quarter(collection_name)
    source_quarter = load_quarter()
    if quarter != source_quarter:
        raise ValueError(
            f"collection quarter {quarter} does not match EBM source {source_quarter}"
        )
    docs = documents(quarter)
    store = open_store(collection_name, force_recreate=True)
    try:
        store.add_documents(docs)
    finally:
        store.client.close()
    return len(docs)


def documents(quarter: str) -> list[Document]:
    return [
        Document(
            page_content=gop.embedding_text,
            metadata={
                "code": gop.code,
                "specialties": list(gop.specialties),
                "quarter": quarter,
            },
        )
        for gop in load_gops()
    ]
