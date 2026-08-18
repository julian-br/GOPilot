"""Vector store over the EBM catalogue."""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.paths import CHROMA

ANY_SPECIALTY = "*"


def open_index(embedding_model: str) -> Chroma:
    """One collection per embedding model — their vectors have different dimensions."""
    return Chroma(
        collection_name=embedding_model.replace(":", "_"),
        embedding_function=OllamaEmbeddings(model=embedding_model),
        persist_directory=str(CHROMA),
    )


def billable_by(specialty: str) -> dict:
    """Codes without a specialty list are billable by anyone, so they must be included."""
    return {"$or": [{"specialties": {"$contains": specialty}},
                    {"specialties": {"$contains": ANY_SPECIALTY}}]}
