from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from src.db.patients import Patient
from src.generation.prompt_inputs import format_patient_context
from src.generation.prompts import RAG_BILLING_PROMPT
from src.generation.schemas import RecommendationResult, RecommendationRun


@dataclass(frozen=True)
class RagPredictor:
    model: BaseChatModel
    retriever: BaseRetriever
    close_resources: Callable[[], None] = lambda: None

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        candidates = self.retriever.invoke(dictation)
        chain = (
            RAG_BILLING_PROMPT
            | self.model.with_structured_output(
                RecommendationResult, include_raw=True
            )
            | RunnableLambda(RecommendationRun.from_output)
        ).with_config({"run_name": "recommendation"})
        return chain.invoke(
            {
                "dictation": dictation,
                "patient_context": format_patient_context(patient),
                "candidates": _candidate_context(candidates),
            }
        )

    def close(self) -> None:
        self.close_resources()


def _candidate_context(candidates: list[Document]) -> str:
    return "\n\n".join(
        f"{candidate.metadata['code']}\n{candidate.page_content}"
        for candidate in candidates
    )
