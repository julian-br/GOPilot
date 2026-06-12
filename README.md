# GOPilot

GOPilot is a local AI assistant for EBM billing code recommendation in German GP practices. Given a doctor's dictation and patient context, it recommends GOP (Gebührenordnungsposition) codes from the EBM 2026 catalogue.

It runs fully offline using a local LLM (Qwen3.5 9B via Ollama) and a local vector database (ChromaDB). Built for research and evaluation, not production use.

## How it works

Three recommendation modes, compared in the evaluation:

- **Basic** — plain LLM with patient context
- **RAG** — LLM + top-k semantic retrieval from ChromaDB
- **Agent** — HyDE document + neutral search terms → hybrid retrieval (semantic + BM25) → cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) → structured decision step constrained to the retrieved candidates

There is no hard-coded GOP knowledge: every recommended code must come out of retrieval. Deterministic filters only encode catalogue rules (age limits from the GOP text, Kapitel-III specialty restriction), nothing tuned to the test cases.

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and a running [Ollama](https://ollama.com/).

```bash
conda env create -f environment.yml
conda activate gopilot
python setup.py   # pulls models, inits DB, fetches + ingests EBM PDF, caches reranker
```

## Usage

```bash
python chat.py            # interactive agent chat
python -m src.eval        # evaluate all conditions -> reports/default.json
python -m src.fetch_ebm   # re-fetch + re-ingest latest EBM (e.g. quarterly update)
```

## Evaluation

20 hand-annotated GP billing cases (`data/test_dictations/`), scored with per-case F1 against ground-truth GOPs. Predicted codes that are already billed in the quarter are removed before scoring — uniformly in all conditions, mirroring practice management software. Results land in `reports/<experiment>.json`.

Current results (avg F1, 20 cases, 3 runs): **basic 0.10 · RAG 0.33 · agent 0.60** — zero variance across runs (GPU inference at `temperature=0` is deterministic here; verify with `python -m src.eval --runs 3`).

Caveat: the test cases were also used while iterating on prompts and retrieval, so reported numbers are dev-set numbers. An unbiased estimate needs newly written, held-out cases.
