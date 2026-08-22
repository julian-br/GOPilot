"""The annotated test dictations."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.paths import DATA
from src.patient import Patient, PreviousQuarterContact

CASES = DATA / "test_dictations"


@dataclass(frozen=True)
class Case:
    case_id: str
    dictation: str
    patient: Patient
    expected: tuple[str, ...]


def load_cases(path: Path = CASES) -> list[Case]:
    return [_case(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(path.glob("case_*.json"))]


def _case(raw: dict) -> Case:
    patient = raw["patient"]
    return Case(
        case_id=raw["case_id"],
        dictation=raw["dictation"],
        patient=Patient(
            id=patient["id"],
            age=patient["age"],
            gender=patient["gender"],
            insurance=patient["insurance"],
            conditions=tuple(patient["conditions"]),
            billed_gops_current_quarter=tuple(
                patient["billed_gops_current_quarter"]
            ),
            previous_quarter_contacts=tuple(
                PreviousQuarterContact(
                    quarter=contact["quarter"],
                    contact_type=contact["contact_type"],
                    reason=contact["reason"],
                )
                for contact in patient["previous_quarter_contacts"]
            ),
        ),
        expected=tuple(raw["expected_gops"]),
    )
