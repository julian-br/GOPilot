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
XML — 3,570 GOPs plus 3,354 regional suffix variants, with points, exclusions, base-service
dependencies, age limits, quantity limits and the obligatory service content as structured fields.

The data is free of charge but licensed for use, not redistribution (KBV master data licence, §3).
It stays out of version control — see `.gitignore`. Only code is published here.

## Setup

```bash
conda env create -f environment.yml
conda activate gopilot
python setup.py
```
