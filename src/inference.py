"""
LLM inference layer: prompts, call, and GOP extraction from model output.
"""

import json
import re

import ollama

MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = """\
Du bist ein EBM-Abrechnungsexperte. Analysiere das Arztdiktat und nenne die \
korrekten GOP-Nummern (5-stellig) aus dem EBM 2026, die abgerechnet werden können.

Antworte NUR mit einer JSON-Liste der GOP-Nummern, z.B.: ["03000", "03321"].
Wenn keine Leistung abrechenbar ist, antworte: []
Keine Erklärungen, kein Markdown, keine Alternativen, nur die JSON-Liste."""

_PROMPT_PATIENT = """\
Patient: {patient_context}

Diktat:
{dictation}"""

_PROMPT_RAG = """\
Patient: {patient_context}

Relevante GOPs aus dem EBM (Kandidaten aus der Datenbank, nicht alle müssen passen):
{gop_context}

Wähle nur GOPs aus den oben genannten Kandidaten, wenn sie zum Diktat und Patientenkontext passen.

Diktat:
{dictation}"""


def build_prompt(
    dictation: str,
    patient_context: str,
    gop_context: str | None = None,
) -> str:
    if gop_context and patient_context:
        return _PROMPT_RAG.format(
            patient_context=patient_context,
            gop_context=gop_context,
            dictation=dictation,
        )
    return _PROMPT_PATIENT.format(
        patient_context=patient_context,
        dictation=dictation,
    )


def ask_llm(user: str, system: str = SYSTEM_PROMPT, model: str = MODEL, think: bool = False) -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0} if think else {"temperature": 0, "num_predict": 100},
        think=think,
    )
    return resp.message.content.strip()


def parse_gops(text: str) -> list[str]:
    """Extract 5-digit GOP numbers from LLM response.

    Tries all JSON arrays from last to first — the agent always ends with its
    final recommendation as the last [...], so this prefers that over intermediate
    arrays that appear earlier in the response.
    """
    matches = list(re.finditer(r"\[.*?\]", text, re.DOTALL))
    for m in reversed(matches):
        try:
            candidates = json.loads(m.group())
            gops = _unique(str(g).strip() for g in candidates if re.fullmatch(r"\d{5}", str(g).strip()))
            if gops or candidates == []:
                return gops
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
