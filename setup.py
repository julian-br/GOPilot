"""One-shot setup: download the EBM catalogue and build the index."""

from scripts import download_ebm
from src.config import RetrievalConfig, load_config
from src.ingest import build_index

if __name__ == "__main__":
    config = load_config()
    if not isinstance(config, RetrievalConfig):
        raise ValueError("setup requires a rag or agent configuration")
    download_ebm.main()
    print(
        f"  indexed {build_index(config.ebm_collection):,} GOPs"
    )
