from operator import itemgetter
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src.generation.prompt_inputs import format_candidates, format_patient_context
from src.generation.prompts import RAG_BILLING_PROMPT
from src.generation.schemas import RecommendationResult
from src.patient import Patient


class RagPredictor:
    def __init__(
        self,
        model: BaseChatModel,
        retriever: BaseRetriever,
        quarter: str,
        close_resources: Callable[[], None] = lambda: None,
    ) -> None:
        retrieval = (
            RunnableLambda(itemgetter("dictation"), name="extract_dictation")
            | retriever.with_config({"run_name": "retrieve_gops"})
            | RunnableLambda(format_candidates, name="format_candidates")
        )
        self._chain: Runnable = (
            RunnablePassthrough.assign(candidates=retrieval)
            | RAG_BILLING_PROMPT
            | model.with_structured_output(
                RecommendationResult, method="function_calling"
            )
        ).with_config({"run_name": "recommendation"})
        self._quarter = quarter
        self._close_resources = close_resources

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationResult:
        return self._chain.invoke(
            {
                "dictation": dictation,
                "quarter": self._quarter,
                "patient_context": format_patient_context(
                    patient, current_quarter=self._quarter
                ),
            }
        )

    def close(self) -> None:
        self._close_resources()
