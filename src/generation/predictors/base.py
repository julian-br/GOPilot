from typing import Protocol

from src.generation.schemas import RecommendationRun
from src.patient import Patient


class Predictor(Protocol):
    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        ...

    def close(self) -> None:
        ...
