"""Vector store over the EBM catalogue."""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import Config, load_config
from src.paths import CHROMA

ANY_SPECIALTY = "*"


def open_index(config: Config | None = None) -> Chroma:
    """One collection per embedding model — their vectors have different dimensions."""
    config = config or load_config()
    return Chroma(
        collection_name=config.embedding_model.replace(":", "_"),
        embedding_function=OllamaEmbeddings(model=config.embedding_model),
        persist_directory=str(CHROMA),
    )


def billable_by(specialty: str) -> dict:
    """Codes without a specialty list are billable by anyone, so they must be included."""
    return {"$or": [{"specialties": {"$contains": specialty}},
                    {"specialties": {"$contains": ANY_SPECIALTY}}]}
