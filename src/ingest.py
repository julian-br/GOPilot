"""Fill the vector store with the EBM catalogue."""

from langchain_core.documents import Document

from src.config import Config
from src.db.vectors import open_index
from src.ebm import GOP, load_gops


def build_index(config: Config | None = None) -> int:
    gops = load_gops()
    index = open_index(config)
    index.add_documents([_document(g) for g in gops], ids=[g.code for g in gops])
    return len(gops)


def _document(gop: GOP) -> Document:
    return Document(
        page_content=gop.embedding_text,
        metadata={
            "code": gop.code,
            "primary_care": gop.billable_in("1"),
            "specialist": gop.billable_in("2"),
        },
    )
