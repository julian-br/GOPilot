"""
GOPilot agent: tool-calling loop that maps a dictation + patient context to EBM GOPs.
"""

import json
from datetime import date

import chromadb
import ollama

from src.db import get_patient_context
from src.ingest import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, OllamaEmbedder

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


def get_patient(patient_id: str) -> dict | None:
    """Retrieve patient context including already-billed GOPs this quarter."""
    today = date.today()
    quartal = f"{(today.month - 1) // 3 + 1}/{today.year}"
    return get_patient_context(patient_id, quartal)


def calculate_euro(punkte: int) -> float:
    """Convert Punkte to Euro using the 2026 Orientierungspunktwert."""
    return round(punkte * ORIENTIERUNGSPUNKTWERT, 2)


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

SYSTEM_PROMPT = """Du bist GOPilot, ein KI-Assistent für die EBM-Abrechnung in deutschen Arztpraxen.

Deine Aufgabe: Analysiere Arztdiktate und empfehle die korrekten GOP-Nummern aus dem EBM 2026.

Vorgehensweise:
1. Hole zuerst den Patientenkontext (get_patient), falls eine Patienten-ID genannt wird.
2. Suche für jede genannte Leistung nach passenden GOPs (search_gops).
3. Prüfe Details und Ausschlüsse für relevante GOPs (get_gop_details).
4. Berücksichtige: bereits abgerechnete GOPs, Alter/Geschlecht, Ausschlüsse.
5. Gib eine klare Empfehlung mit GOP-Nummern, Beschreibung und Punkten aus.

Antworte immer auf Deutsch. Sei präzise und erkläre deine Entscheidungen kurz."""


def run_agent(user_message: str, patient_id: str | None = None) -> str:
    """Run one agent turn: process a dictation and return billing recommendations."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.message

        if not msg.tool_calls:
            return msg.content

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments)
            fn = TOOL_FNS.get(fn_name)
            if fn is None:
                result = f"Unknown tool: {fn_name}"
            else:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = f"Error: {e}"

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })
