from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.generation.prompt_inputs import format_patient_context
from src.generation.prompts import BILLING_WITHOUT_RETRIEVAL_PROMPT
from src.generation.schemas import RecommendationResult
from src.patient import Patient


class NoRagPredictor:
    def __init__(self, model: BaseChatModel, quarter: str) -> None:
        self._chain: Runnable = (
            BILLING_WITHOUT_RETRIEVAL_PROMPT
            | model.with_structured_output(
                RecommendationResult, method="function_calling"
            )
        ).with_config({"run_name": "recommendation"})
        self._quarter = quarter

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
        pass
