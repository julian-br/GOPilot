# GOPilot

GOPilot is a local AI assistant for EBM billing code recommendation in German GP practices. Given a
doctor's dictation and patient context, it recommends GOP (Gebührenordnungsposition) codes from the
EBM catalogue.

GOPilot is an exploratory research project for understanding how local language models and EBM
catalogue retrieval can support billing-code recommendations. It is intended for experimentation
and evaluation, not production use.

## Data source

The EBM catalogue comes from the KBV **SDEBM** master data feed, not from the PDF:
[update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/](https://update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/)

`python setup.py` downloads the installer jar and extracts record type 850 (the nationwide EBM) as
XML — 3,570 GOPs (regional suffix variants excluded), with points, exclusions, base-service
dependencies, age limits and the obligatory service content as structured fields.

Setup writes the catalogue to the versioned Qdrant collection configured by `ebm_collection`.
RAG and agent runs derive their catalogue quarter from the collection name (for example,
`ebm_2026_q4`); `no_rag` declares it explicitly. The embedding model is intentionally fixed in
code, while `no_rag` needs no vector-store settings.

Design decisions are logged in [docs/decisions.md](docs/decisions.md).

## Setup

```bash
conda env create -f environment.yml
conda activate gopilot
python setup.py
```

## MLflow

Start the local MLflow dashboard from the repository root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in the browser.

## Evaluation

Each test case is an independent snapshot. Its patient history is fixed input and is never updated
with recommendations from another case. `expected_gops` contains GOPs the physician should actively
submit or recommend; GOPs added automatically by the KV are outside the evaluation scope.

### Preliminary results

The following micro-averaged GOP-level results are a preliminary baseline on the 20 local test
dictations, using `gemma4:e4b` and the current prompts and catalogue configuration. They are for
comparison during development, not a claim of clinical or billing accuracy.

| Strategy | Precision | Recall |    F1 | Error rate |
| -------- | --------: | -----: | ----: | ---------: |
| No RAG   |     0.161 |  0.135 | 0.147 |      0.000 |
| RAG      |     0.381 |  0.216 | 0.276 |      0.050 |
| Agent    |     0.308 |  0.216 | 0.254 |      0.000 |

The RAG run had one structured-output validation error; the other 19 cases completed. All 20
agent cases completed without an execution error.

### Cloud-model comparison

The following runs use the same 20 test dictations, catalogue collection and prompts. They are
also preliminary and measure agreement with the current `expected_gops` labels only.

| Model                                | Strategy | Precision | Recall |    F1 | Error rate |
| ------------------------------------ | -------- | --------: | -----: | ----: | ---------: |
| `stealth/ox-alpha`                   | Agent    |     0.811 |  0.811 | 0.811 |      0.000 |
| `nvidia/nemotron-3.5-lightning:free` | Agent    |     0.577 |  0.405 | 0.476 |      0.000 |
| `nvidia/nemotron-3.5-lightning:free` | RAG      |     0.800 |  0.324 | 0.462 |      0.000 |

### Possible future directions

- GraphRAG could be interesting.
- A stricter workflow could be beneficial.
- HyDE could be useful.
- Although the agent can query individual services, it has limited information about their billing
  rules. The relevant information exists in the EBM but is not retrieved reliably yet (for example,
  exclusions and dependencies).
- Automatic rule checking could help.
