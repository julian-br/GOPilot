from langchain_core.language_models.chat_models import BaseChatModel

from src.db.patients import Patient
from src.generation.prompts import BILLING_WITHOUT_RETRIEVAL_PROMPT
from src.generation.schemas import RecommendationResult


def recommend_without_retrieval(
    model: BaseChatModel, dictation: str, patient: Patient | None
) -> RecommendationResult:
    chain = BILLING_WITHOUT_RETRIEVAL_PROMPT | model.with_structured_output(RecommendationResult)
    result = chain.invoke(
        {
            "dictation": dictation,
            "patient_context": _patient_context(patient),
        }
    )
    return RecommendationResult(
        recommendations=[
            recommendation
            for recommendation in result.recommendations
            if recommendation.code.isdigit() and len(recommendation.code) == 5
        ]
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
