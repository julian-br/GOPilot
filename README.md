# GOPilot

GOPilot is a local AI assistant for EBM billing code recommendation in German GP practices. Given a
doctor's dictation and patient context, it recommends GOP (Gebührenordnungsposition) codes from the
EBM catalogue.

GOPilot is an exploratory project for understanding how language models and EBM catalogue
retrieval can support billing-code recommendations. The project is currently being refactored around
LangChain and remains under active development. It is intended for experimentation and evaluation,
rather than production use.

An earlier, more prescriptive workflow achieved promising results on a small evaluation set, but it
was too biased towards anticipated billing codes. The current iteration rebuilds the pipeline around
less prescriptive retrieval, agent behaviour and evaluation.

## Data source

The EBM catalogue comes from the KBV **SDEBM** master data feed, not from the PDF:
[update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/](https://update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/)

`python setup.py` downloads the installer jar and extracts record type 850 (the nationwide EBM) as
XML — 3,570 GOPs (regional suffix variants excluded), with points, exclusions, base-service
dependencies, age limits and the obligatory service content as structured fields.

Setup writes the catalogue to the versioned Qdrant collection configured by `ebm_collection`.
RAG and agent runs derive their catalogue quarter from the collection name (for example,
`ebm_2026_q4`).

## Current predictors

The following predictors were implemented after the LangChain refactor:

- **No RAG:** a baseline with no retrieval and no catalogue context; it recommends codes only from
  the dictation and patient context.
- **RAG:** retrieves a fixed set of catalogue candidates before the model makes a recommendation.
- **Agent:** lets the model search and inspect individual catalogue entries through tools before it
  recommends codes.

## Setup

```bash
conda env create -f environment.yml
conda activate gopilot
python setup.py
```

## Running evaluations

Each test case is an independent snapshot. Its patient history is fixed input and is never updated
with recommendations from another case. `expected_gops` contains GOPs the system should
submit or recommend.

Select a configuration from `configs/` to run a generation evaluation:

```bash
# Replace experiment.yaml with the configuration to evaluate
python -m src.eval.generation --config configs/experiment.yaml

# Evaluate retrieval coverage independently of generation
python -m src.eval.retrieval --config configs/experiment.yaml
```

## MLflow

MLflow is used for experiment tracking and tracing.

Start the local MLflow dashboard from the repository root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in the browser.

## Evaluation results

### Preliminary results

The following results are a preliminary baseline on the 20 local test
dictations, using `gemma4:e4b` and the current prompts and catalogue configuration. They are for
comparison during development.

| Strategy | Precision | Recall |    F1 | Error rate |
| -------- | --------: | -----: | ----: | ---------: |
| No RAG   |     0.161 |  0.135 | 0.147 |      0.000 |
| RAG      |     0.381 |  0.216 | 0.276 |      0.050 |
| Agent    |     0.308 |  0.216 | 0.254 |      0.000 |

The RAG run had one structured-output validation error; the other 19 cases completed. All 20
agent cases completed without an execution error. Overall, these results are currently weak. The
agent often finds relevant GOPs through custom search queries, but then either omits them from its
final recommendation or selects unrelated candidates instead.

### Cloud-model comparison

The following runs use the same 20 test dictations, catalogue collection and prompts. Their purpose
is to better understand how much performance depends on model capability. The more capable cloud
model performs substantially better, but still falls short of a reliable solution.

| Model                                | Strategy | Precision | Recall |    F1 | Error rate |
| ------------------------------------ | -------- | --------: | -----: | ----: | ---------: |
| `stealth/ox-alpha`                   | Agent    |     0.811 |  0.811 | 0.811 |      0.000 |
| `nvidia/nemotron-3.5-lightning:free` | Agent    |     0.577 |  0.405 | 0.476 |      0.000 |

### Possible future directions

- GraphRAG could be interesting.
- Reimplement a more explicit workflow.
- Improve retrieval, for example with HyDE.
- Although the agent can query individual services, it has limited information about their billing
  rules. The relevant information exists in the EBM but is not retrieved reliably yet (for example,
  exclusions and dependencies).
- Automatic rule checking could help.
