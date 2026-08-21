from typing import Protocol

from src.db.patients import Patient
from src.generation.schemas import RecommendationRun


class Predictor(Protocol):
    def predict(self, dictation: str, patient: Patient | None) -> RecommendationRun:
        ...

    def close(self) -> None:
        ...
