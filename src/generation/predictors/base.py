from typing import Protocol

from src.generation.schemas import RecommendationResult
from src.patient import Patient


class Predictor(Protocol):
    def predict(self, dictation: str, patient: Patient | None) -> RecommendationResult:
        ...

    def close(self) -> None:
        ...
