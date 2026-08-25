"""The EBM catalogue as documents, and the vector index built from them."""

from dataclasses import asdict

from langchain_core.documents import Document

from src.db.vectors import collection_quarter, open_store
from src.ebm import GOP, load_gops, load_quarter


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
        store.add_documents(docs, batch_size=16)
    finally:
        store.client.close()
    return len(docs)


def documents(quarter: str) -> list[Document]:
    return [
        Document(
            page_content=gop.embedding_text,
            metadata={
                "code": gop.code,
                "code_type": gop.code_type,
                "annotations": list(gop.annotations),
                "billing_rules": _billing_rules(gop),
                "specialties": list(gop.specialties),
                "quarter": quarter,
            },
        )
        for gop in load_gops()
    ]


def _billing_rules(gop: GOP) -> dict[str, object]:
    rules: dict[str, object] = {}
    if gop.billing_text:
        rules["text"] = gop.billing_text
    if gop.occurrence_limits:
        rules["occurrence_limits"] = [
            asdict(limit) for limit in gop.occurrence_limits
        ]
    return rules
