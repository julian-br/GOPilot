# GOPilot

GOPilot is a local AI assistant for EBM billing code recommendation in German GP practices. Given a
doctor's dictation and patient context, it recommends GOP (Gebührenordnungsposition) codes from the
EBM catalogue.

Built for research and evaluation, not production use. Currently being rebuilt from scratch — this
README grows back as the pipeline does.

## Data source

The EBM catalogue comes from the KBV **SDEBM** master data feed, not from the PDF:
[update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/](https://update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/)

`python setup.py` downloads the installer jar and extracts record type 850 (the nationwide EBM) as
XML — 3,570 GOPs (regional suffix variants excluded), with points, exclusions, base-service
dependencies, age limits and the obligatory service content as structured fields.

Setup writes the catalogue to the versioned Qdrant collection configured by `ebm_collection`.
Runtime derives the catalogue quarter from that collection name and never accesses the source XML.

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
