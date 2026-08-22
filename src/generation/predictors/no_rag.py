from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from src.db.patients import Patient
from src.generation.prompt_inputs import format_patient_context
from src.generation.prompts import BILLING_WITHOUT_RETRIEVAL_PROMPT
from src.generation.schemas import RecommendationResult, RecommendationRun


@dataclass(frozen=True)
class NoRagPredictor:
    model: BaseChatModel

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        chain = (
            BILLING_WITHOUT_RETRIEVAL_PROMPT
            | self.model.with_structured_output(
                RecommendationResult, include_raw=True
            )
            | RunnableLambda(RecommendationRun.from_output)
        ).with_config({"run_name": "recommendation"})
        return chain.invoke(
            {
                "dictation": dictation,
                "patient_context": format_patient_context(patient),
            }
        )

    def close(self) -> None:
        pass
