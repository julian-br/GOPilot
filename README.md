# GOPilot

Local EBM billing assistant based on Qwen3.5 9B.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Ollama](https://ollama.com/) installed and running

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate gopilot

# 2. Run one-shot setup (pulls models, inits DB, fetches + ingests EBM data)
python setup.py
```

That's it. Setup pulls `qwen3.5:9b` and `nomic-embed-text` via Ollama, downloads
the latest KBV EBM PDF, parses all GOPs and stores them in ChromaDB.

## Usage

```bash
# Chat with the model
python chat.py

# Re-fetch and re-ingest latest EBM (e.g. after a quarterly update)
python -m src.fetch_ebm

# Re-ingest already downloaded PDF
python -m src.fetch_ebm --ingest-only
```

## Project structure

```
GOPilot/
├── setup.py                   # one-shot setup script
├── chat.py                    # interactive chat
├── src/
│   ├── db.py                  # SQLite mock patient DB
│   ├── fetch_ebm.py           # fetch latest KBV EBM PDF
│   └── ingest.py              # PDF parser + ChromaDB ingest
├── data/
│   ├── ebm_raw/               # downloaded EBM PDFs
│   ├── chroma_db/             # vector DB (generated)
│   ├── gopilot.db             # SQLite patient DB (generated)
│   └── test_dictations/       # test cases with ground truth GOPs
├── environment.yml
└── pyproject.toml
```
