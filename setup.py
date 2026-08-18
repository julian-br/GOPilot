"""One-shot setup: download the EBM catalogue, seed the patient database, build the index."""

from scripts import download_ebm, seed_patients
from src.config import load_config
from src.ingest import build_index

if __name__ == "__main__":
    config = load_config()
    download_ebm.main()
    seed_patients.main()
    print(f"  indexed {build_index(config.embedding_model):,} GOPs")
