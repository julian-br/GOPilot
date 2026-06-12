"""
One-shot setup: pull Ollama models, init DB, fetch EBM PDF, ingest into ChromaDB.

Usage:
    python setup.py
    python setup.py --skip-models   # skip ollama pulls (already done)
    python setup.py --skip-ingest   # skip PDF fetch + ingest
"""

import argparse
import subprocess
import sys

import ollama

from src.agent import MODEL, RERANKER_MODEL
from src.db import SEED_PATIENTS, init_db, seed_db
from src.fetch_ebm import fetch_latest_pdf
from src.ingest import EMBED_MODEL, ingest

REQUIRED_MODELS = [MODEL, EMBED_MODEL]


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def pull_models() -> None:
    step("Pulling Ollama models")
    local = {m.model for m in ollama.list().models}
    for model in REQUIRED_MODELS:
        if model in local:
            print(f"  {model}: already present, skipping")
            continue
        print(f"  Pulling {model} ...")
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode != 0:
            print(f"ERROR: Failed to pull {model}. Is Ollama running?", file=sys.stderr)
            sys.exit(1)
        print(f"  {model}: done")


def pull_reranker() -> None:
    step(f"Caching reranker model ({RERANKER_MODEL})")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("  sentence-transformers not installed — agent will fall back to lexical reranking")
        return
    CrossEncoder(RERANKER_MODEL, trust_remote_code=True)
    print("  Reranker cached locally")


def setup_db() -> None:
    step("Initializing SQLite database with mock patients")
    init_db()
    seed_db()
    print(f"  DB ready at data/gopilot.db ({len(SEED_PATIENTS)} mock patients)")


def setup_vectordb() -> None:
    step("Fetching latest EBM PDF and ingesting into ChromaDB")
    pdf_path = fetch_latest_pdf()
    ingest(pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GOPilot one-shot setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip ollama pull")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip EBM fetch + ingest")
    args = parser.parse_args()

    print("\nGOPilot Setup")
    print("Prerequisite: Ollama must be running (ollama serve or Ollama app)\n")

    if not args.skip_models:
        pull_models()
        pull_reranker()

    setup_db()

    if not args.skip_ingest:
        setup_vectordb()

    step("Setup complete")
    print("  Run 'python chat.py' to start chatting with the model.")
    print("  Run 'python -m src.eval' to evaluate the current test dictations.")
    print("  Run 'python -m src.fetch_ebm' to re-fetch and re-ingest EBM data.")


if __name__ == "__main__":
    main()
