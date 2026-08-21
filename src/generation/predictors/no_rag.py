from dataclasses import dataclass

import mlflow
from langchain_core.language_models.chat_models import BaseChatModel
from mlflow.entities import SpanType

from src.db.patients import Patient
from src.generation.prompts import BILLING_WITHOUT_RETRIEVAL_PROMPT
from src.generation.schemas import RecommendationResult, RecommendationRun


@dataclass(frozen=True)
class NoRagPredictor:
    model: BaseChatModel

    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        chain = BILLING_WITHOUT_RETRIEVAL_PROMPT | self.model.with_structured_output(
            RecommendationResult, include_raw=True
        )
        result = chain.invoke(
            {
                "dictation": dictation,
                "patient_context": _patient_context(patient),
            }
        )
        if result["parsing_error"] is not None:
            raise result["parsing_error"]

        reasoning = result["raw"].additional_kwargs.get("reasoning_content")
        if reasoning:
            with mlflow.start_span("model_reasoning", span_type=SpanType.LLM) as span:
                span.set_outputs(reasoning)

        return RecommendationRun(
            result=RecommendationResult(
                recommendations=[
                    recommendation
                    for recommendation in result["parsed"].recommendations
                    if recommendation.code.isdigit() and len(recommendation.code) == 5
                ]
            ),
            reasoning=reasoning,
        )


def _patient_context(patient: Patient | None) -> str:
    if patient is None:
        return "Nicht verfuegbar."
    return "\n".join(
        [
            f"Alter: {patient.age}",
            f"Geschlecht: {patient.gender}",
            f"Versicherung: {patient.insurance}",
            f"Bekannte Diagnosen: {', '.join(patient.conditions) or 'keine'}",
            f"Bereits in diesem Quartal abgerechnet: {', '.join(patient.billed_gops) or 'keine'}",
            f"Erster Kontakt im Quartal: {'ja' if patient.first_contact else 'nein'}",
        ]
    )
