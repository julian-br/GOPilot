"""Fill the vector store with the EBM catalogue."""

from langchain_core.documents import Document

from src.db.vectors import ANY_SPECIALTY, open_index
from src.ebm import GOP, load_gops


def build_index(embedding_model: str) -> int:
    gops = load_gops()
    index = open_index(embedding_model)
    index.add_documents([_document(g) for g in gops], ids=[g.code for g in gops])
    return len(gops)


def _document(gop: GOP) -> Document:
    return Document(
        page_content=gop.embedding_text,
        metadata={"code": gop.code, "specialties": list(gop.specialties) or [ANY_SPECIALTY]},
    )
