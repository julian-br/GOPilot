import sqlite3
from pathlib import Path

DB_PATH = Path("data/gopilot.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    birth_year  INTEGER NOT NULL,
    gender      TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    insurance   TEXT NOT NULL,
    conditions  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS quarterly_billing (
    patient_id  TEXT NOT NULL REFERENCES patients(id),
    quartal     TEXT NOT NULL,
    gop         TEXT NOT NULL,
    PRIMARY KEY (patient_id, quartal, gop)
);
"""

SEED_PATIENTS = [
    ("P001", "Müller, Hans", 1959, "M", "AOK Bayern", "Diabetes mellitus Typ 2; arterielle Hypertonie"),
    ("P002", "Schmidt, Anna", 1992, "F", "TK", ""),
    ("P003", "Weber, Erika", 1948, "F", "Barmer", "COPD; Herzinsuffizienz; arterielle Hypertonie"),
    ("P004", "Becker, Thomas", 1976, "M", "DAK-Gesundheit", ""),
    ("P005", "Kaya, Leyla", 1985, "F", "AOK Nordost", "Asthma bronchiale"),
    ("P006", "Sommer, Karl", 1948, "M", "KNAPPSCHAFT", "Herzrhythmusstörungen; koronare Herzkrankheit"),
    ("P007", "Fischer, Marie", 2005, "F", "BKK Freudenberg", ""),
]

# GOPs already billed in Q2/2026 for realistic same-quarter follow-up contexts.
SEED_BILLING = [
    ("P001", "2/2026", "03000"),
    ("P001", "2/2026", "03220"),
    ("P002", "2/2026", "03000"),
    ("P003", "2/2026", "03000"),
    ("P004", "2/2026", "03000"),
]


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _ensure_columns(conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
    if "conditions" not in columns:
        conn.execute("ALTER TABLE patients ADD COLUMN conditions TEXT NOT NULL DEFAULT ''")


def seed_db() -> None:
    with get_connection() as conn:
        seed_ids = [p[0] for p in SEED_PATIENTS]
        conn.executemany(
            "DELETE FROM quarterly_billing WHERE patient_id = ?", [(pid,) for pid in seed_ids]
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO patients
                (id, name, birth_year, gender, insurance, conditions)
            VALUES (?,?,?,?,?,?)
            """,
            SEED_PATIENTS,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO quarterly_billing VALUES (?,?,?)", SEED_BILLING
        )


def get_patient_context(patient_id: str, quartal: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        if row is None:
            return None

        billed = conn.execute(
            "SELECT gop FROM quarterly_billing WHERE patient_id = ? AND quartal = ?",
            (patient_id, quartal),
        ).fetchall()

        billed_gops = [r["gop"] for r in billed]
        return {
            "patient_id": patient_id,
            "name": row["name"],
            "age": 2026 - row["birth_year"],
            "gender": row["gender"],
            "insurance": row["insurance"],
            "conditions": [c.strip() for c in row["conditions"].split(";") if c.strip()],
            "quartal": quartal,
            "first_contact_this_quarter": len(billed_gops) == 0,
            "gops_already_billed": billed_gops,
        }


def main() -> None:
    init_db()
    seed_db()
    print(f"Database initialized at {DB_PATH}")
    for pid in ("P001", "P002", "P003", "P004", "P005"):
        ctx = get_patient_context(pid, "2/2026")
        print(f"  {pid}: {ctx}")


if __name__ == "__main__":
    main()
