"""Create the mock patient database from data/patients_seed.json."""

import json
import sqlite3

from src.db.patients import SCHEMA
from src.paths import DATA, PATIENT_DB

SEED = DATA / "patients_seed.json"


def main() -> None:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    PATIENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PATIENT_DB) as con:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT OR REPLACE INTO patients VALUES (:id,:name,:birth_year,:gender,:insurance,:conditions)",
            seed["patients"])
        con.executemany(
            "INSERT OR REPLACE INTO billed VALUES (:patient_id,:quarter,:gop)",
            seed["billed"])
    print(f"  {PATIENT_DB}: {len(seed['patients'])} patients, {len(seed['billed'])} billed entries")


if __name__ == "__main__":
    main()
