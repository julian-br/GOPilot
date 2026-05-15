"""
GOPilot agent: tool-calling loop that maps a dictation + patient context to EBM GOPs.
"""

import json
from datetime import date

import chromadb
import ollama

from src.db import get_patient_context
from src.ingest import CHROMA_PATH, COLLECTION_NAME, OllamaEmbedder

MODEL = "qwen3.5:9b"
ORIENTIERUNGSPUNKTWERT = 0.127404  # Euro per Punkt, 2026

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbedder(),
        )
    return _collection


# --- Tool implementations ---

def search_gops(query: str, n_results: int = 5) -> list[dict]:
    """Semantic search over ChromaDB GOP descriptions."""
    results = _get_collection().query(query_texts=[query], n_results=n_results)
    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        hits.append({
            "gop": meta["gop"],
            "punkte": meta["punkte"],
            "kapitel": meta["kapitel"],
            "fachgruppe": meta["fachgruppe"],
            "document": results["documents"][0][i],
        })
    return hits


def get_gop_details(gop: str) -> dict | None:
    """Fetch a single GOP entry by code."""
    results = _get_collection().get(ids=[f"gop_{gop}"], include=["documents", "metadatas"])
    if not results["ids"]:
        return None
    meta = results["metadatas"][0]
    return {
        "gop": meta["gop"],
        "punkte": meta["punkte"],
        "kapitel": meta["kapitel"],
        "fachgruppe": meta["fachgruppe"],
        "ausschluesse": json.loads(meta.get("ausschluesse", "[]")),
        "document": results["documents"][0],
    }


def get_patient(patient_id: str, quartal: str | None = None) -> dict | None:
    """Retrieve patient context including already-billed GOPs this quarter."""
    if quartal is None:
        today = date.today()
        quartal = f"{(today.month - 1) // 3 + 1}/{today.year}"
    return get_patient_context(patient_id, quartal)


def calculate_euro(punkte: int) -> float:
    """Convert Punkte to Euro using the 2026 Orientierungspunktwert."""
    return round(punkte * ORIENTIERUNGSPUNKTWERT, 2)


def _patient_summary(ctx: dict | None, patient_id: str, quartal: str) -> str:
    if ctx is None:
        return f"Patient {patient_id} (unbekannt), Quartal {quartal}, bereits abgerechnet: unbekannt"
    already = ", ".join(ctx["gops_already_billed"]) or "keine"
    return (
        f"{ctx['name']}, {ctx['age']} Jahre, {ctx['gender']}, {ctx['insurance']}, "
        f"Quartal {quartal}, bereits abgerechnet: {already}"
    )


# --- Tool schema for Ollama ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_gops",
            "description": (
                "Semantic search for EBM billing codes (GOPs) by medical description. "
                "Use this to find relevant GOPs for procedures mentioned in a dictation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical procedure or symptom to search for"},
                    "n_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gop_details",
            "description": "Get full details for a specific GOP by its 5-digit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gop": {"type": "string", "description": "5-digit GOP code, e.g. '03321'"},
                },
                "required": ["gop"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient",
            "description": (
                "Get patient context: age, gender, insurance, and which GOPs are already billed "
                "this quarter. Always call this before recommending Pauschalen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient ID, e.g. 'P001'"},
                    "quartal": {"type": "string", "description": "Quarter in format Q/YYYY, e.g. '2/2026'"},
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_euro",
            "description": "Convert EBM Punkte to Euro using the 2026 Orientierungspunktwert (12.7404 ct/Punkt).",
            "parameters": {
                "type": "object",
                "properties": {
                    "punkte": {"type": "integer", "description": "Number of Punkte"},
                },
                "required": ["punkte"],
            },
        },
    },
]

TOOL_FNS = {
    "search_gops": search_gops,
    "get_gop_details": get_gop_details,
    "get_patient": get_patient,
    "calculate_euro": calculate_euro,
}

SYSTEM_PROMPT = """\
Du bist GOPilot, ein KI-Assistent für die EBM-Abrechnung in deutschen Arztpraxen.

Deine Aufgabe: Analysiere Arztdiktate und empfehle die korrekten GOP-Nummern aus dem EBM 2026.

Vorgehensweise:
1. Hole zuerst den Patientenkontext (get_patient) mit der genannten Patienten-ID und dem Quartal.
2. Identifiziere alle im Diktat genannten Leistungen. Suche für jede Leistung mit präzisen \
medizinischen Fachbegriffen (z.B. "Spirometrie", "Ruhe-EKG", "Versichertenpauschale Hausarzt").
3. Prüfe Details und Ausschlüsse für vielversprechende GOPs (get_gop_details).
4. Berücksichtige: bereits abgerechnete GOPs, Alter/Geschlecht/Versicherung, Ausschlüsse.

WICHTIGE REGELN:
- Nenne NUR GOPs, die du explizit über search_gops oder get_gop_details gefunden hast.
- Erfinde KEINE GOP-Nummern. Wenn du keine passende GOP findest, gib [] zurück.
- Schließe IMMER mit einer JSON-Liste ab (letzte Zeile): ["12345", "67890"] oder []

Erkläre kurz deine Entscheidungen, dann als letzte Zeile die JSON-Liste.\
"""


def run_agent(
    dictation: str,
    patient_id: str,
    quartal: str,
    model: str = MODEL,
    think: bool = False,
    max_steps: int = 10,
    already_billed_gops: list[str] | None = None,
) -> dict:
    """Run the agent loop for one dictation. Returns response text and tool log."""
    db_ctx = get_patient_context(patient_id, quartal)
    if db_ctx is not None and already_billed_gops is not None:
        db_ctx = {**db_ctx, "gops_already_billed": already_billed_gops}
    patient_ctx = _patient_summary(db_ctx, patient_id, quartal)

    user_message = (
        f"Patient-ID: {patient_id}, Quartal: {quartal}\n"
        f"Patientenkontext: {patient_ctx}\n\n"
        f"Diktat:\n{dictation}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_log = []
    options = {"temperature": 0, "num_predict": 300}

    for step in range(max_steps):
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
            options=options,
            think=think,
        )
        msg = response.message

        if not msg.tool_calls:
            return {"response": msg.content.strip(), "steps": step + 1, "tool_log": tool_log}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls,
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments)
            fn = TOOL_FNS.get(fn_name)
            result = f"Unknown tool: {fn_name}" if fn is None else _call_tool(fn, fn_args)
            tool_log.append({"tool": fn_name, "args": fn_args, "result": result})
            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })

    return {"response": "max_steps reached", "steps": max_steps, "tool_log": tool_log}


def _call_tool(fn, args: dict):
    try:
        return fn(**args)
    except Exception as e:
        return f"Error: {e}"
