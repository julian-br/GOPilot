"""The annotated test dictations."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.paths import DATA
from src.patient import Patient, PriorContact

CASES = DATA / "test_dictations"


@dataclass(frozen=True)
class Case:
    case_id: str
    quarter: str
    dictation: str
    patient: Patient
    expected: tuple[str, ...]


def load_cases(path: Path = CASES) -> list[Case]:
    return [_case(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(path.glob("case_*.json"))]


def require_catalogue_quarter(cases: list[Case], catalogue_quarter: str) -> None:
    mismatched = [case.case_id for case in cases if case.quarter != catalogue_quarter]
    if mismatched:
        joined_ids = ", ".join(mismatched)
        raise ValueError(
            f"cases use a different EBM quarter than {catalogue_quarter}: {joined_ids}"
        )


def _case(raw: dict) -> Case:
    patient = raw["patient"]
    return Case(
        case_id=raw["case_id"],
        quarter=raw["quartal"],
        dictation=raw["dictation"],
        patient=Patient(
            id=patient["id"],
            age=patient["age"],
            gender=patient["gender"],
            insurance=patient["insurance"],
            conditions=tuple(patient["conditions"]),
            prior_contacts=tuple(
                PriorContact(
                    quarter=contact["quarter"],
                    contact_type=contact["contact_type"],
                    reason=contact["reason"],
                    billed_gops=tuple(contact["billed_gops"]),
                )
                for contact in patient["prior_contacts"]
            ),
        ),
        expected=tuple(raw["expected_gops"]),
    )
