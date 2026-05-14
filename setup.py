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
from pathlib import Path

import ollama

from src.db import init_db, seed_db
from src.fetch_ebm import fetch_latest_pdf
from src.ingest import ingest

REQUIRED_MODELS = ["qwen3.5:9b", "qwen3-embedding:4b"]


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def pull_models() -> None:
    step("Pulling Ollama models")
    local = {m.model for m in ollama.list().models}
    for model in REQUIRED_MODELS:
        # ollama list returns names like "qwen3.5:9b", match by prefix
        already = any(m.startswith(model.split(":")[0]) and model in m for m in local)
        if already:
            print(f"  {model}: already present, skipping")
            continue
        print(f"  Pulling {model} ...")
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode != 0:
            print(f"ERROR: Failed to pull {model}. Is Ollama running?", file=sys.stderr)
            sys.exit(1)
        print(f"  {model}: done")


def setup_db() -> None:
    step("Initializing SQLite database with mock patients")
    init_db()
    seed_db()
    print("  DB ready at data/gopilot.db")
    print("  Mock patients: P001 Müller (67, Chroniker), P002 Schmidt (34), P003 Wagner (8, Kind)")


def setup_vectordb() -> None:
    step("Fetching latest EBM PDF and ingesting into ChromaDB")
    chroma_path = Path("data/chroma_db")
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

    setup_db()

    if not args.skip_ingest:
        setup_vectordb()

    step("Setup complete")
    print("  Run 'python chat.py' to start chatting with the model.")
    print("  Run 'python -m src.fetch_ebm' to re-fetch and re-ingest EBM data.")


if __name__ == "__main__":
    main()
