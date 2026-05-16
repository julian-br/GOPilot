"""
GOPilot agent: tool-calling loop that maps a dictation + patient context to EBM GOPs.
"""

import json
import math
import re
import time
from collections import Counter
from datetime import date

import chromadb
import ollama

from src.db import get_patient_context
from src.ingest import CHROMA_PATH, COLLECTION_NAME, OllamaEmbedder

MODEL = "qwen3.5:9b"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
ORIENTIERUNGSPUNKTWERT = 0.127404  # Euro per Punkt, 2026

_collection = None
_bm25_index = None
_reranker = None
_reranker_model_name = None
_reranker_error = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbedder(),
        )
    return _collection


def _get_bm25_index() -> dict:
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    data = _get_collection().get(include=["documents", "metadatas"])
    documents = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]
    tokenized = [_retrieval_tokens(doc) for doc in documents]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    n_docs = len(documents)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(n_docs, 1)
    idf = {
        token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    _bm25_index = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "tokenized": tokenized,
        "term_counts": [Counter(tokens) for tokens in tokenized],
        "doc_lengths": [len(tokens) for tokens in tokenized],
        "avgdl": avgdl,
        "idf": idf,
    }
    return _bm25_index


def _get_reranker(model_name: str):
    global _reranker, _reranker_model_name, _reranker_error
    if _reranker is not None and _reranker_model_name == model_name:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        _reranker_error = f"sentence-transformers not installed: {e}"
        return None
    try:
        _reranker = CrossEncoder(model_name, trust_remote_code=True, local_files_only=True)
        _reranker_model_name = model_name
        _reranker_error = None
        return _reranker
    except Exception as e:
        _reranker_error = f"failed to load reranker {model_name}: {e}"
        return None


# --- Tool implementations ---

def search_gops(query: str, n_results: int = 8, preferred_fachgruppe: str | None = None) -> list[dict]:
    """Hybrid semantic + lexical search over GOP descriptions."""
    expanded_query = _expand_query(query)
    fetch_n = max(n_results * 4, 50)
    semantic_results = _get_collection().query(
        query_texts=[expanded_query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = []
    for i, doc_id in enumerate(semantic_results["ids"][0]):
        semantic_hits.append(
            _hit_from_meta(
                semantic_results["metadatas"][0][i],
                semantic_results["documents"][0][i],
                distance=semantic_results["distances"][0][i],
                semantic_rank=i + 1,
            )
        )

    bm25_hits = _bm25_search(expanded_query, fetch_n)
    return _reciprocal_rank_fusion(
        semantic_hits=semantic_hits,
        bm25_hits=bm25_hits,
        n_results=n_results,
        preferred_fachgruppe=preferred_fachgruppe,
    )


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


def _hit_from_meta(
    meta: dict,
    document: str,
    distance: float | None = None,
    semantic_rank: int | None = None,
    bm25_rank: int | None = None,
    bm25_score: float | None = None,
) -> dict:
    return {
        "gop": meta["gop"],
        "punkte": meta["punkte"],
        "kapitel": meta["kapitel"],
        "fachgruppe": meta["fachgruppe"],
        "distance": distance,
        "semantic_rank": semantic_rank,
        "bm25_rank": bm25_rank,
        "bm25_score": bm25_score,
        "document": document,
    }


def _bm25_search(query: str, n_results: int) -> list[dict]:
    query_tokens = _retrieval_tokens(query)
    if not query_tokens:
        return []

    index = _get_bm25_index()
    k1 = 1.5
    b = 0.75
    scores: list[tuple[float, int]] = []
    for i, counts in enumerate(index["term_counts"]):
        score = 0.0
        doc_len = index["doc_lengths"][i]
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            idf = index["idf"].get(token, 0.0)
            denom = tf + k1 * (1 - b + b * doc_len / max(index["avgdl"], 1e-9))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scores.append((score, i))

    hits = []
    for rank, (score, i) in enumerate(sorted(scores, reverse=True)[:n_results], 1):
        hits.append(
            _hit_from_meta(
                index["metadatas"][i],
                index["documents"][i],
                bm25_rank=rank,
                bm25_score=score,
            )
        )
    return hits


def _reciprocal_rank_fusion(
    semantic_hits: list[dict],
    bm25_hits: list[dict],
    n_results: int,
    preferred_fachgruppe: str | None = None,
    k: int = 60,
) -> list[dict]:
    merged: dict[str, dict] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(semantic_hits, 1):
        gop = hit["gop"]
        merged.setdefault(gop, hit.copy())
        merged[gop]["semantic_rank"] = rank
        merged[gop]["distance"] = hit.get("distance")
        scores[gop] = scores.get(gop, 0.0) + 1 / (k + rank)

    for rank, hit in enumerate(bm25_hits, 1):
        gop = hit["gop"]
        entry = merged.setdefault(gop, hit.copy())
        entry["bm25_rank"] = rank
        entry["bm25_score"] = hit.get("bm25_score")
        scores[gop] = scores.get(gop, 0.0) + 1 / (k + rank)

    def sort_key(gop: str) -> tuple:
        hit = merged[gop]
        return (
            -scores[gop],
            hit.get("semantic_rank") or 9999,
            hit.get("bm25_rank") or 9999,
            gop,
        )

    result = []
    for gop in sorted(merged, key=sort_key)[:n_results]:
        item = merged[gop]
        item["retrieval_score"] = scores[gop]
        result.append(item)
    return result


def get_patient(patient_id: str, quartal: str | None = None) -> dict | None:
    """Retrieve patient context including already-billed GOPs this quarter."""
    if quartal is None:
        today = date.today()
        quartal = f"{(today.month - 1) // 3 + 1}/{today.year}"
    return get_patient_context(patient_id, quartal)


def calculate_euro(punkte: int) -> float:
    """Convert Punkte to Euro using the 2026 Orientierungspunktwert."""
    return round(punkte * ORIENTIERUNGSPUNKTWERT, 2)


def _expand_query(query: str) -> str:
    return query


def build_search_plan(
    dictation: str,
    patient_id: str,
    quartal: str,
    patient_context: dict | None = None,
    practice_fachgruppe: str | None = None,
    model: str = MODEL,
    think: bool = False,
    max_terms: int = 12,
) -> list[str]:
    """Extract neutral EBM search terms from a dictation without choosing GOPs."""
    response = _chat_with_retry(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du extrahierst neutrale Suchbegriffe fuer eine EBM-Recherche. "
                    "Du entscheidest keine GOPs und nennst keine GOP-Nummern. "
                    "Gib ausschliesslich eine JSON-Liste kurzer deutscher Suchbegriffe aus. "
                    "Extrahiere nur Leistungen, Kontakte, Untersuchungen, Diagnostik, Therapien, "
                    "Prozeduren, Verordnungen, Bescheinigungen und dokumentierte Bedingungen, die "
                    "ausdruecklich im Diktat stehen. "
                    "Keine Spekulationen."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient-ID: {patient_id}, Quartal: {quartal}\n\n"
                    f"Abrechnende Praxis/Fachgruppe: {practice_fachgruppe or 'unbekannt'}\n\n"
                    f"Patientenkontext:\n{json.dumps(patient_context, ensure_ascii=False)}\n\n"
                    f"Diktat:\n{dictation}"
                ),
            },
        ],
        options={"temperature": 0, "num_predict": 250},
        think=think,
    )
    return _clean_search_plan((response.message.content or "").strip(), max_terms=max_terms)


def build_hypothetical_document(
    dictation: str,
    patient_id: str,
    quartal: str,
    patient_context: dict | None = None,
    practice_fachgruppe: str | None = None,
    model: str = MODEL,
    think: bool = False,
) -> str:
    """Create a neutral HyDE query document for semantic retrieval."""
    response = _chat_with_retry(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Erstelle ein neutrales hypothetisches EBM-Dokument fuer semantische Suche. "
                    "Das Dokument soll die im Diktat dokumentierten abrechnungsrelevanten "
                    "Kontakte, Leistungen, Prozeduren, Untersuchungen und Randbedingungen in "
                    "normalen Saetzen beschreiben. Nenne keine GOP-Nummern, keine erwarteten "
                    "Ergebnisse und keine Leistungen, die nicht dokumentiert sind. "
                    "Uebernimm relevante Patientenkontexte nur als Kontext, nicht als Diagnose-"
                    "Spekulation. Gib nur das Suchdokument aus."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient-ID: {patient_id}, Quartal: {quartal}\n\n"
                    f"Abrechnende Praxis/Fachgruppe: {practice_fachgruppe or 'unbekannt'}\n\n"
                    f"Patientenkontext:\n{json.dumps(patient_context, ensure_ascii=False)}\n\n"
                    f"Diktat:\n{dictation}"
                ),
            },
        ],
        options={"temperature": 0, "num_predict": 260},
        think=think,
    )
    return " ".join((response.message.content or "").split())


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
                    "query": {"type": "string", "description": "Medical procedure, billing term, or symptom to search for"},
                    "n_results": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
                    "preferred_fachgruppe": {
                        "type": "string",
                        "description": "Optional preferred EBM fachgruppe from the billing practice context.",
                    },
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

Verfügbare Tools: get_patient, search_gops, get_gop_details, calculate_euro. Verwende keine anderen Toolnamen.
Eine neutrale Suchplanung kann im User-Kontext enthalten sein. Sie ist keine Empfehlung und keine GOP-Liste.

Vorgehensweise:
1. Rufe als ersten Schritt das Tool get_patient mit der genannten Patienten-ID und dem Quartal auf.
2. Nutze die Suchplanung und das Diktat, um die relevanten Leistungen und Abrechnungskontexte zu recherchieren.
3. Rufe search_gops mit präzisen medizinischen oder abrechnungsbezogenen Suchbegriffen auf.
4. Prüfe Details und Ausschlüsse für vielversprechende GOPs (get_gop_details).
5. Berücksichtige: bereits abgerechnete GOPs, Alter/Geschlecht/Versicherung, Ausschlüsse.

WICHTIGE REGELN:
- Nenne NUR GOPs, die du explizit über search_gops oder get_gop_details gefunden hast.
- Erfinde KEINE GOP-Nummern. Wenn du keine passende GOP findest, gib [] zurück.
- Beschreibe keine zukünftige Suche. Wenn du suchen willst, rufe sofort das passende Tool auf.
- Wiederhole keine identischen Suchanfragen. Finalisiere, sobald genug Kandidaten geprüft sind.
- Die finale JSON-Liste ist KEINE Kandidatenliste. Nenne nur eindeutig passende GOPs.
- Wenn mehrere Suchtreffer Alternativen für dieselbe Leistung sind, wähle höchstens die passendste GOP.
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
    practice_fachgruppe: str | None = None,
    reranker_model: str | None = RERANKER_MODEL,
    already_billed_gops: list[str] | None = None,
) -> dict:
    """Run a deterministic research pass, then ask the model for a structured decision."""
    tool_log = []

    patient_ctx = get_patient(patient_id, quartal)
    if patient_ctx is not None and already_billed_gops is not None:
        patient_ctx = {**patient_ctx, "gops_already_billed": already_billed_gops}
    tool_log.append({"tool": "get_patient", "args": {"patient_id": patient_id, "quartal": quartal}, "result": patient_ctx})

    try:
        hypothetical_document = build_hypothetical_document(
            dictation=dictation,
            patient_id=patient_id,
            quartal=quartal,
            patient_context=patient_ctx,
            practice_fachgruppe=practice_fachgruppe,
            model=model,
            think=think,
        )
    except ollama.ResponseError as e:
        return {"response": "[]", "steps": 1, "tool_log": tool_log, "search_plan": [], "error": str(e)}

    search_queries = [
        {"label": "hyde_document", "query": hypothetical_document},
        {"label": "dictation", "query": dictation},
    ]
    candidates: dict[str, dict] = {}
    for item in search_queries:
        search_args = {"query": item["query"], "n_results": 16, "preferred_fachgruppe": practice_fachgruppe}
        hits = _call_tool(search_gops, search_args)
        tool_log.append({"tool": "search_gops", "args": search_args, "result": hits})
        if not isinstance(hits, list):
            continue
        for rank, hit in enumerate(hits, 1):
            if not isinstance(hit, dict) or not hit.get("gop"):
                continue
            gop = hit["gop"]
            entry = candidates.setdefault(
                gop,
                {
                    "gop": gop,
                    "source_terms": [],
                    "query_texts": [],
                    "best_rank": rank,
                    "best_distance": hit.get("distance"),
                    "search_hit": hit,
                    "details": None,
                },
            )
            entry["source_terms"].append(item["label"])
            entry["query_texts"].append(item["query"])
            entry["best_rank"] = min(entry["best_rank"], rank)
            if hit.get("distance") is not None:
                entry["best_distance"] = min(entry.get("best_distance") or hit["distance"], hit["distance"])

    rerank_text = f"{dictation}\n\n{hypothetical_document}"
    for gop in _rank_candidate_codes(
        candidates,
        preferred_fachgruppe=practice_fachgruppe,
        query_text=rerank_text,
    )[:28]:
        details = _call_tool(get_gop_details, {"gop": gop})
        tool_log.append({"tool": "get_gop_details", "args": {"gop": gop}, "result": details})
        if isinstance(details, dict):
            candidates[gop]["details"] = details

    rerank_log = rerank_candidates(
        query_text=rerank_text,
        patient_ctx=patient_ctx,
        candidates=candidates,
        model_name=reranker_model,
    )

    already_billed = set(patient_ctx.get("gops_already_billed", [])) if patient_ctx else set()
    ranked_codes = [
        gop
        for gop in _rank_candidate_codes(
            candidates,
            preferred_fachgruppe=practice_fachgruppe,
            already_billed_gops=already_billed,
            query_text=rerank_text,
        )
        if gop not in already_billed and _unsupported_special_context_penalty(candidates[gop]) == 0
        and _patient_context_mismatch_penalty(candidates[gop], patient_ctx) == 0
        and not _is_low_information_candidate(candidates[gop])
    ]

    response = _decide_from_candidates(
        model=model,
        dictation=dictation,
        patient_id=patient_id,
        quartal=quartal,
        patient_ctx=patient_ctx,
        practice_fachgruppe=practice_fachgruppe,
        search_plan=[hypothetical_document],
        candidates=[candidates[gop] for gop in ranked_codes[:24]],
        think=think,
    )
    return {
        "response": response,
        "steps": 2,
        "tool_log": tool_log,
        "search_plan": [hypothetical_document],
        "hypothetical_document": hypothetical_document,
        "reranker": rerank_log,
    }


def _call_tool(fn, args: dict):
    try:
        return fn(**args)
    except Exception as e:
        return f"Error: {e}"


def _rank_candidate_codes(
    candidates: dict[str, dict],
    preferred_fachgruppe: str | None = None,
    already_billed_gops: set[str] | None = None,
    query_text: str = "",
) -> list[str]:
    return sorted(
        candidates,
        key=lambda gop: (
            gop in (already_billed_gops or set()),
            -float(candidates[gop].get("rerank_score", -999.0)),
            _fachgruppe_priority(candidates[gop], preferred_fachgruppe),
            _unsupported_special_context_penalty(candidates[gop]),
            -_lexical_overlap(query_text, candidates[gop]),
            candidates[gop].get("best_distance", 99.0) or 99.0,
            candidates[gop].get("best_rank", 99),
            -len(set(candidates[gop].get("source_terms", []))),
            gop,
        ),
    )


def rerank_candidates(
    query_text: str,
    patient_ctx: dict | None,
    candidates: dict[str, dict],
    model_name: str | None = RERANKER_MODEL,
) -> dict:
    rerankable_codes = [
        gop
        for gop, candidate in candidates.items()
        if isinstance(candidate.get("details"), dict)
        and _patient_context_mismatch_penalty(candidate, patient_ctx) == 0
        and not _is_low_information_candidate(candidate)
    ]
    if not rerankable_codes:
        return {"model": model_name, "used": False, "error": "no rerankable candidates", "items": []}

    if not model_name:
        return {"model": None, "used": False, "error": "reranker disabled", "items": []}

    reranker = _get_reranker(model_name)
    if reranker is None:
        _assign_fallback_rerank_scores(query_text, candidates, rerankable_codes)
        return {
            "model": model_name,
            "used": False,
            "error": _reranker_error,
            "items": _rerank_items_for_report(candidates, rerankable_codes),
        }

    pairs = [
        [query_text, _candidate_document(candidates[gop])]
        for gop in rerankable_codes
    ]
    try:
        scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    except Exception as e:
        _assign_fallback_rerank_scores(query_text, candidates, rerankable_codes)
        return {
            "model": model_name,
            "used": False,
            "error": f"reranker predict failed: {e}",
            "items": _rerank_items_for_report(candidates, rerankable_codes),
        }

    for gop, score in zip(rerankable_codes, scores):
        candidates[gop]["rerank_score"] = float(score)
        candidates[gop]["rerank_method"] = "cross_encoder"

    return {
        "model": model_name,
        "used": True,
        "error": None,
        "items": _rerank_items_for_report(candidates, rerankable_codes),
    }


def _assign_fallback_rerank_scores(query_text: str, candidates: dict[str, dict], codes: list[str]) -> None:
    for gop in codes:
        candidates[gop]["rerank_score"] = _lexical_overlap(query_text, candidates[gop])
        candidates[gop]["rerank_method"] = "lexical_fallback"


def _candidate_document(candidate: dict) -> str:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    return _truncate(str(details.get("document") or hit.get("document") or ""), 1800)


def _rerank_items_for_report(candidates: dict[str, dict], codes: list[str]) -> list[dict]:
    ranked = sorted(codes, key=lambda gop: float(candidates[gop].get("rerank_score", -999.0)), reverse=True)
    return [
        {
            "gop": gop,
            "score": candidates[gop].get("rerank_score"),
            "method": candidates[gop].get("rerank_method"),
            "best_rank": candidates[gop].get("best_rank"),
            "fachgruppe": (
                candidates[gop].get("details", {}).get("fachgruppe")
                if isinstance(candidates[gop].get("details"), dict)
                else None
            ),
        }
        for gop in ranked[:30]
    ]


def _fachgruppe_priority(candidate: dict, preferred_fachgruppe: str | None) -> int:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    fachgruppe = str(details.get("fachgruppe") or hit.get("fachgruppe") or "")
    if preferred_fachgruppe and fachgruppe == preferred_fachgruppe:
        return 0
    if "Arztgruppenübergreifende" in fachgruppe:
        return 1
    return 2


def _lexical_overlap(query_text: str, candidate: dict) -> float:
    if not query_text:
        return 0.0
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    document = str(details.get("document") or hit.get("document") or "")
    query_tokens = _content_tokens(query_text)
    doc_tokens = _content_tokens(document)
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _retrieval_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-zäöüßA-ZÄÖÜ0-9]{2,}", text.casefold())
        if token not in _stopwords()
    ]


def _normalize_token(token: str) -> str:
    token = (
        token.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    for suffix in ("innen", "licher", "ische", "ungen", "ung", "chen", "lich", "eren", "erer", "ere", "ern", "em", "en", "er", "es", "e", "s", "n"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _stopwords() -> set[str]:
    return {
        "aber",
        "auch",
        "auf",
        "aus",
        "bei",
        "das",
        "dem",
        "den",
        "der",
        "des",
        "die",
        "ein",
        "eine",
        "einer",
        "eines",
        "fuer",
        "für",
        "im",
        "in",
        "ist",
        "mit",
        "nach",
        "oder",
        "und",
        "von",
        "zum",
        "zur",
    }


def _content_tokens(text: str) -> set[str]:
    return {token for token in _retrieval_tokens(text) if len(token) >= 4}


def _unsupported_special_context_penalty(candidate: dict) -> int:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    document = str(details.get("document") or hit.get("document") or "").casefold()
    source = " ".join([*candidate.get("source_terms", []), *candidate.get("query_texts", [])]).casefold()
    special_markers = [
        "unvorhergesehener inanspruchnahme",
        "zwischen 19:00 und 7:00",
        "samstagen",
        "sonntagen",
        "feiertagen",
        "notdienst",
        "besuch",
        "visite",
        "empfängnisregelung",
        "sterilisation",
        "schwangerschaftsabbruch",
        "krankenkasse",
        "muster 50",
        "großen gelenkes",
        "grossen gelenkes",
        "unelastischer",
        "palliativ",
        "geriatr",
        "nichtärztlich",
        "nichtaerztlich",
        "häuslichkeit",
        "haeuslichkeit",
        "chronische erkrankung",
        "belastungs",
        "langzeit-ekg",
    ]
    if not any(marker in document for marker in special_markers):
        return 0
    if any(
        marker in source
        for marker in (
            "unvorhergesehen",
            "19:00",
            "samstag",
            "sonntag",
            "feiertag",
            "notdienst",
            "besuch",
            "visite",
            "empfängnisregelung",
            "sterilisation",
            "schwangerschaftsabbruch",
            "krankenkasse",
            "muster 50",
            "großes gelenk",
            "grosses gelenk",
            "unelastisch",
            "palliativ",
            "geriatr",
            "nichtärztlich",
            "nichtaerztlich",
            "häuslichkeit",
            "haeuslichkeit",
            "chronisch",
            "chronische erkrankung",
            "belastung",
            "langzeit-ekg",
        )
    ):
        return 0
    return 1


def _patient_context_mismatch_penalty(candidate: dict, patient_ctx: dict | None) -> int:
    if not patient_ctx:
        return 0
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    document = str(details.get("document") or hit.get("document") or "").casefold()
    fachgruppe = str(details.get("fachgruppe") or hit.get("fachgruppe") or "")
    age = patient_ctx.get("age")
    if isinstance(age, int):
        for match in re.finditer(r"bis zum vollendeten\s+(\d+)", document):
            if age > int(match.group(1)):
                return 1
        if age >= 18 and "Kinder- und Jugendmedizin" in fachgruppe:
            return 1
    return 0


def _is_low_information_candidate(candidate: dict) -> bool:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    document = str(details.get("document") or hit.get("document") or "")
    if len(document) >= 140:
        return False
    return not any(marker in document for marker in ("Obligater Leistungsinhalt", "Punkte", "Euro"))


def _decide_from_candidates(
    model: str,
    dictation: str,
    patient_id: str,
    quartal: str,
    patient_ctx: dict | None,
    practice_fachgruppe: str | None,
    search_plan: list[str],
    candidates: list[dict],
    think: bool,
) -> str:
    candidate_summary = [_candidate_for_prompt(candidate) for candidate in candidates[:14]]
    if not candidate_summary:
        return "[]"
    allowed_gops = [candidate["gop"] for candidate in candidate_summary]
    try:
        response = _chat_with_retry(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein EBM-Abrechnungsexperte. Entscheide aus einer vorgegebenen "
                        "Kandidatenliste, welche GOPs zum Diktat und Patientenkontext passen. "
                        "Die Kandidaten stammen aus search_gops/get_gop_details. Du darfst keine "
                        "anderen GOPs nennen. Suchtreffer sind nur Kandidaten, nicht automatisch "
                        "korrekt. Waehle eine GOP nur, wenn ihr obligater Leistungsinhalt im Diktat "
                        "dokumentiert ist oder sich direkt aus Patientenkontakt und Kontext ergibt. "
                        "Bewerte jeden Kandidaten unabhaengig. Verwirf nicht alle Kandidaten, nur "
                        "weil einige Suchtreffer unpassend sind. Wenn die Beschreibung eines "
                        "Kandidaten eine dokumentierte Untersuchung, Prozedur oder Therapie direkt "
                        "abdeckt, soll dieser Kandidat ausgewaehlt werden, sofern keine klare "
                        "Gegenbedingung in Details oder Patientenkontext erkennbar ist. Nutze die "
                        "source_terms als Hinweis, welche dokumentierte Leistung den Kandidaten "
                        "gefunden hat. Wenn source_terms und Kandidatendokument dieselbe Leistung "
                        "fachsprachlich beschreiben, waehle den Kandidaten. Warte nicht auf absolute "
                        "Sicherheit, wenn der Leistungsinhalt dokumentiert ist. "
                        "Bereits in diesem Quartal abgerechnete GOPs wurden aus der erlaubten "
                        "Kandidatenliste entfernt. Noch nicht abgerechnete Zuschlaege duerfen "
                        "trotzdem gewaehlt werden, wenn der obligate Leistungsinhalt dokumentiert "
                        "ist. Eine Versichertenpauschale kann direkt durch einen persoenlichen "
                        "oder Video-Arzt-Patienten-Kontakt im Quartal begruendet sein. Ein "
                        "Chronikerzuschlag kann direkt durch chronische Erkrankung, persoenlichen "
                        "Kontakt und dokumentierte fortlaufende Betreuung begruendet sein. "
                        "Eine Diagnose, ein Symptom oder ein aehnlicher Begriff allein reicht nicht. "
                        "Beruecksichtige Alter, Fachgruppe und dokumentierte Umstaende: Kinder- und "
                        "Jugendmedizin passt nur bei Kindern/Jugendlichen oder passendem Kontext; "
                        "Bevorzuge bei gleichwertigen Kandidaten die abrechnende Praxis/Fachgruppe. "
                        "Kandidaten anderer Fachgruppen sollen nur gewaehlt werden, wenn sie fachlich "
                        "klar besser passen oder arztgruppenuebergreifend sind. "
                        "Randzeiten, Notdienst, palliativmedizinische Versorgung, spezielle Programme "
                        "oder Bescheinigungen passen nur, wenn sie dokumentiert sind. Wenn ein "
                        "Kandidatentext Sonderbedingungen wie unvorhergesehene Inanspruchnahme, "
                        "bestimmte Uhrzeiten, Wochenende/Feiertag, organisierter Notdienst, "
                        "Besuch/Visite oder spezielle Versorgung nennt, waehle ihn nur bei expliziter "
                        "Dokumentation dieser Sonderbedingung. Bei fehlender Sonderbedingung bevorzuge "
                        "allgemeinere Kandidaten fuer dieselbe Leistung, falls vorhanden. "
                        "Unterscheide einfache Messungen von speziellen Langzeit-, Belastungs-, "
                        "apparativen oder komplexen Untersuchungen: Waehle solche Spezialkandidaten "
                        "nur, wenn Dauer, Belastung, Langzeitaufzeichnung, Apparateeinsatz oder "
                        "Komplexinhalt passend dokumentiert ist. "
                        "Gib ausschliesslich eine JSON-Liste der final empfohlenen 5-stelligen "
                        "GOP-Strings aus. Wenn kein Kandidat plausibel passt, gib [] aus."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Patient-ID: {patient_id}, Quartal: {quartal}\n"
                        f"Abrechnende Praxis/Fachgruppe: {practice_fachgruppe or 'unbekannt'}\n"
                        f"Patientenkontext:\n{json.dumps(patient_ctx, ensure_ascii=False)}\n\n"
                        f"Neutrale Suchplanung:\n{json.dumps(search_plan, ensure_ascii=False)}\n\n"
                        f"Diktat:\n{dictation}\n\n"
                        f"Erlaubte GOPs:\n{json.dumps(allowed_gops, ensure_ascii=False)}\n\n"
                        f"Kandidaten:\n{json.dumps(candidate_summary, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            options={"temperature": 0, "num_predict": 180},
            think=think,
        )
    except ollama.ResponseError:
        return "[]"
    return (response.message.content or "").strip()


def _candidate_for_prompt(candidate: dict) -> dict:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    return {
        "gop": candidate.get("gop"),
        "source_terms": list(dict.fromkeys(candidate.get("source_terms", []))),
        "best_rank": candidate.get("best_rank"),
        "punkte": details.get("punkte", hit.get("punkte")),
        "fachgruppe": details.get("fachgruppe", hit.get("fachgruppe")),
        "ausschluesse": details.get("ausschluesse", []),
        "document": _truncate(str(details.get("document") or hit.get("document", "")), 600),
    }


def _filter_decision_response(text: str, allowed_gops: list[str]) -> str:
    allowed = set(allowed_gops)
    selected = []
    for gop in _parse_json_list(text):
        code = str(gop).strip()
        if code in allowed and code not in selected:
            selected.append(code)
    return json.dumps(selected, ensure_ascii=False)


def _chat_with_retry(*args, retries: int = 2, **kwargs):
    for attempt in range(retries + 1):
        try:
            return ollama.chat(*args, **kwargs)
        except ollama.ResponseError:
            if attempt >= retries:
                raise
            time.sleep(2)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _clean_search_plan(text: str, max_terms: int) -> list[str]:
    values = _parse_json_list(text)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = " ".join(value.strip().split())
        if not term or len(term) > 120 or _looks_like_gop(term):
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            result.append(term)
        if len(result) >= max_terms:
            break
    return result


def _parse_json_list(text: str) -> list:
    matches = []
    start = text.find("[")
    while start != -1:
        end = text.find("]", start)
        if end == -1:
            break
        matches.append(text[start:end + 1])
        start = text.find("[", end + 1)
    for raw in reversed(matches):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def _looks_like_gop(term: str) -> bool:
    compact = term.strip().replace(" ", "")
    return compact.isdigit() and len(compact) == 5
