# GOPilot

Local EBM billing assistant based on Qwen3.5 9B.

## Prerequisites

Conda and Ollama installed.

## Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate gopilot

# 2. Pull model (runs on GPU automatically)
ollama pull qwen3.5:9b

# 3. Start
python chat.py
```
