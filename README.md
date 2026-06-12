# GOPilot

GOPilot is a local AI assistant for EBM billing code recommendation in German GP practices. Given a doctor's dictation and patient context, it recommends the correct GOP (Gebührenordnungsposition) codes from the EBM 2026 catalogue.

It runs fully offline using a local LLM (Qwen3.5 9B via Ollama) and a local vector database (ChromaDB). The system is built for research and evaluation, not production use.

## How it works

GOPilot offers three recommendation modes, evaluated against a ground-truth test set:

| Mode | Description | Avg F1 (20 cases) |
|------|-------------|-------------------|
| **Basic** | Plain LLM with patient context | 0.20 |
| **RAG** | LLM + top-k semantic retrieval from ChromaDB | 0.51 |
| **Agent** | Multi-query retrieval + BGE reranker + structured decision step | **0.67** |

The agent pipeline:
1. Builds a hypothetical EBM document (HyDE) and extracts neutral search terms from the dictation
2. Runs up to 8 parallel hybrid searches (semantic + BM25) against the EBM vector DB
3. Injects fundamental Pauschalen (03000, 03220) that semantic search misses when specific procedures dominate
4. Reranks all candidates with `BAAI/bge-reranker-v2-m3` (cross-encoder, raw logits)
5. Passes the top-24 ranked candidates to a structured decision model that selects only what is explicitly documented

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

Setup pulls `qwen3.5:9b` and `qwen3-embedding:4b` via Ollama, downloads the latest KBV EBM PDF, parses all GOPs, and stores them in ChromaDB.

## Usage

```bash
# Interactive agent chat
python chat.py

# Run evaluation across all conditions and write reports/default.json
python -m src.eval

# Run evaluation with a custom config
python -m src.eval --config configs/default.yaml --verbose

# Re-fetch and re-ingest latest EBM (e.g. after a quarterly update)
python -m src.fetch_ebm

# Re-ingest an already downloaded PDF
python -m src.fetch_ebm --ingest-only
```

## Project structure

```
GOPilot/
├── setup.py                   # one-shot setup script
├── chat.py                    # interactive agent chat
├── configs/
│   └── default.yaml           # evaluation config (model, conditions, reranker)
├── src/
│   ├── agent.py               # core agent: retrieval, reranking, decision
│   ├── db.py                  # SQLite mock patient DB with seed data
│   ├── eval.py                # evaluation runner (basic / rag / agent)
│   ├── fetch_ebm.py           # fetch latest KBV EBM PDF
│   ├── inference.py           # LLM prompt builder + GOP list parser
│   └── ingest.py              # EBM PDF parser + ChromaDB ingest
├── data/
│   ├── ebm_raw/               # downloaded EBM PDFs
│   ├── chroma_db/             # vector DB (generated)
│   ├── gopilot.db             # SQLite patient DB (generated)
│   └── test_dictations/       # 20 annotated test cases with ground truth GOPs
├── reports/
│   └── default.json           # latest evaluation report
└── environment.yml
```

## Evaluation

The test set contains 20 hand-crafted cases covering common GP billing scenarios: first contacts, chronic patient follow-ups, home visits, phone consultations, EKG, spirometry, wound care, and others. Each case has a dictation, patient context (already billed GOPs, diagnoses), and ground truth GOPs.

Cases where the expected answer is `[]` (no billable service) count as correct when the agent also returns `[]`.

Note: The LLM runs at `temperature=0` but CPU inference is not bit-exact, so individual case results vary slightly between runs. Average F1 is stable within ±0.05 across runs.
