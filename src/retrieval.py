"""Find candidate GOPs for a dictation."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.db.vectors import billable_by


def find_candidates(index: Chroma, dictation: str, specialty: str, k: int) -> list[Document]:
    return index.similarity_search(dictation, k=k, filter=billable_by(specialty))
