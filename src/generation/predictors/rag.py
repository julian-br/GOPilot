from dataclasses import dataclass
from typing import Callable

import mlflow
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from mlflow.entities import SpanType

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
        chain = RAG_BILLING_PROMPT | self.model.with_structured_output(
            RecommendationResult, include_raw=True
        )
        result = chain.invoke(
            {
                "dictation": dictation,
                "patient_context": format_patient_context(patient),
                "candidates": _candidate_context(candidates),
            }
        )
        if result["parsing_error"] is not None:
            raise result["parsing_error"]

        reasoning = result["raw"].additional_kwargs.get("reasoning_content")
        if reasoning:
            with mlflow.start_span("model_reasoning", span_type=SpanType.LLM) as span:
                span.set_outputs(reasoning)

        return RecommendationRun(
            result=result["parsed"],
            reasoning=reasoning,
        )

    def close(self) -> None:
        self.close_resources()


def _candidate_context(candidates: list[Document]) -> str:
    return "\n\n".join(
        f"{rank}. GOP {candidate.metadata['code']}\n{candidate.page_content}"
        for rank, candidate in enumerate(candidates, 1)
    )
