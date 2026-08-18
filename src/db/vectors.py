"""Vector store over the EBM catalogue."""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import Config, load_config
from src.paths import CHROMA


def open_index(config: Config | None = None) -> Chroma:
    """One collection per embedding model — their vectors have different dimensions."""
    config = config or load_config()
    return Chroma(
        collection_name=config.embedding_model.replace(":", "_"),
        embedding_function=OllamaEmbeddings(model=config.embedding_model),
        persist_directory=str(CHROMA),
    )
