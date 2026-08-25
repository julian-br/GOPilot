from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EBM = DATA / "ebm"
EBM_MASTER = EBM / "master.xml"
EBM_KEYTABS = EBM / "keytabs"
QDRANT = DATA / "qdrant"
FASTEMBED = DATA / "fastembed"
