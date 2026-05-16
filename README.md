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

That's it. Setup pulls `qwen3.5:9b` and `qwen3-embedding:4b` via Ollama, downloads
the latest KBV EBM PDF, parses all GOPs and stores them in ChromaDB.

## Usage

```bash
# Chat with the model
python chat.py

# Evaluate all configured conditions and write reports/default.json
python -m src.eval

# Re-fetch and re-ingest latest EBM (e.g. after a quarterly update)
python -m src.fetch_ebm

# Re-ingest already downloaded PDF
python -m src.fetch_ebm --ingest-only
```

## Project structure

```
GOPilot/
├── setup.py                   # one-shot setup script
├── chat.py                    # interactive agent chat
├── configs/
│   └── default.yaml           # evaluation conditions
├── src/
│   ├── agent.py               # tool-capable billing agent
│   ├── db.py                  # SQLite mock patient DB
│   ├── eval.py                # evaluation runner
│   ├── fetch_ebm.py           # fetch latest KBV EBM PDF
│   ├── inference.py           # LLM prompt + GOP parsing
│   └── ingest.py              # PDF parser + ChromaDB ingest
├── data/
│   ├── ebm_raw/               # downloaded EBM PDFs
│   ├── chroma_db/             # vector DB (generated)
│   ├── gopilot.db             # SQLite patient DB (generated)
│   └── test_dictations/       # test cases with ground truth GOPs
├── reports/
│   └── default.json           # latest evaluation report
└── environment.yml
```

### TODO

Real test cases. In the best case we would use real (anomynized) medical data, eventuell aus MFA schulungsdaten

Maybe better embedding model: jina-embeddings-v3 oder BGE-M3

BM25 (Keyword-Search) und Vector-Search mit Reranking

Hypothetical Document Embeddings für bessere queries?

erst zerlegen zu lassen: "Welche medizinischen Konzepte stecken in dieser Behandlung?
