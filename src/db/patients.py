"""Mock practice management system: patient master data and what was billed this quarter."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.paths import PATIENT_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    birth_year INTEGER NOT NULL,
    gender     TEXT NOT NULL,
    insurance  TEXT NOT NULL,
    conditions TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS billed (
    patient_id TEXT NOT NULL,
    quarter    TEXT NOT NULL,
    gop        TEXT NOT NULL,
    PRIMARY KEY (patient_id, quarter, gop)
);
"""


@dataclass(frozen=True)
class Patient:
    """`age` is the year of the quarter minus the birth year, ignoring the birthday."""

    id: str
    name: str
    age: int
    gender: str
    insurance: str
    conditions: tuple[str, ...]
    billed_gops: tuple[str, ...]

    @property
    def first_contact(self) -> bool:
        return not self.billed_gops


def get_patient(patient_id: str, quarter: str, path: Path = PATIENT_DB) -> Patient | None:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if row is None:
            return None
        billed = tuple(g for (g,) in con.execute(
            "SELECT gop FROM billed WHERE patient_id = ? AND quarter = ? ORDER BY gop",
            (patient_id, quarter)))
    return Patient(
        id=row["id"],
        name=row["name"],
        age=int(quarter.split("/")[1]) - row["birth_year"],
        gender=row["gender"],
        insurance=row["insurance"],
        conditions=tuple(c.strip() for c in row["conditions"].split(";") if c.strip()),
        billed_gops=billed,
    )
