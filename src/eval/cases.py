"""The annotated test dictations."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.paths import DATA

CASES = DATA / "test_dictations"


@dataclass(frozen=True)
class Case:
    case_id: str
    dictation: str
    patient_id: str
    quarter: str
    expected: tuple[str, ...]


def load_cases(path: Path = CASES) -> list[Case]:
    return [_case(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(path.glob("case_*.json"))]


def _case(raw: dict) -> Case:
    return Case(
        case_id=raw["case_id"],
        dictation=raw["dictation"],
        patient_id=raw["patient_id"],
        quarter=raw["quartal"],
        expected=tuple(raw["expected_gops"]),
    )
