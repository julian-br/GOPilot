from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from src.generation.prompt_inputs import format_candidates, format_patient_context
from src.generation.prompts import RAG_BILLING_PROMPT
from src.generation.schemas import RecommendationResult, RecommendationRun
from src.patient import Patient


@dataclass(frozen=True)
class RagPredictor:
    model: BaseChatModel
    retriever: BaseRetriever
    quarter: str
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
                "quarter": self.quarter,
                "patient_context": format_patient_context(patient),
                "candidates": format_candidates(candidates),
            }
        )

    def close(self) -> None:
        self.close_resources()
