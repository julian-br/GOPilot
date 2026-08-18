from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EBM = DATA / "ebm"
EBM_MASTER = EBM / "master.xml"
PATIENT_DB = DATA / "patients.db"
CHROMA = DATA / "chroma"
