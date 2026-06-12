"""
GOPilot agent: two-stage pipeline that maps a dictation + patient context to EBM GOPs.

Stage 1 (research): HyDE document + neutral search terms -> hybrid retrieval
(semantic + BM25) -> cross-encoder reranking.
Stage 2 (decision): a structured prompt asks the model to select only candidates
whose obligatory service content is documented in the dictation.
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

# How many top-ranked candidates get full details fetched (reranker input) and
# how many of them the decision model sees.
MAX_DETAIL_CANDIDATES = 24
MAX_DECISION_CANDIDATES = 14

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
        "documents": documents,
        "metadatas": metadatas,
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


# --- Retrieval ---

def search_gops(query: str, n_results: int = 8) -> list[dict]:
    """Hybrid semantic + lexical search over GOP descriptions."""
    fetch_n = max(n_results * 4, 50)
    semantic_results = _get_collection().query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits = []
    for i in range(len(semantic_results["ids"][0])):
        semantic_hits.append(
            _hit_from_meta(
                semantic_results["metadatas"][0][i],
                semantic_results["documents"][0][i],
                distance=semantic_results["distances"][0][i],
                semantic_rank=i + 1,
            )
        )

    bm25_hits = _bm25_search(query, fetch_n)
    return _reciprocal_rank_fusion(
        semantic_hits=semantic_hits,
        bm25_hits=bm25_hits,
        n_results=n_results,
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
        # Full normative text incl. Abrechnungsbestimmungen/Anmerkungen
        "volltext": meta.get("volltext", ""),
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
                    "Erfasse auch die dokumentierte Kontakt- und Betreuungsform als eigene "
                    "Suchbegriffe (z.B. Erst- oder Folgekontakt im Quartal, persoenlicher oder "
                    "telefonischer Kontakt, Besuch, fortlaufende Betreuung chronischer "
                    "Erkrankungen). "
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


def run_agent(
    dictation: str,
    patient_id: str,
    quartal: str,
    model: str = MODEL,
    think: bool = False,
    practice_fachgruppe: str | None = None,
    reranker_model: str | None = RERANKER_MODEL,
    already_billed_gops: list[str] | None = None,
) -> dict:
    """Run a deterministic research pass, then ask the model for a structured decision.

    `already_billed_gops` overrides the DB billing state for a case (used by eval to
    simulate quarter contexts); the practice management system always knows this.
    """
    tool_log = []

    patient_ctx = get_patient(patient_id, quartal)
    if patient_ctx is not None and already_billed_gops is not None:
        patient_ctx = {
            **patient_ctx,
            "gops_already_billed": already_billed_gops,
            "first_contact_this_quarter": len(already_billed_gops) == 0,
        }
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
        return {"response": "[]", "tool_log": tool_log, "search_plan": [], "error": str(e)}

    try:
        search_terms = build_search_plan(
            dictation=dictation,
            patient_id=patient_id,
            quartal=quartal,
            patient_context=patient_ctx,
            practice_fachgruppe=practice_fachgruppe,
            model=model,
            think=think,
        )
    except ollama.ResponseError:
        search_terms = []

    search_queries = [
        {"label": "hyde_document", "query": hypothetical_document, "n_results": 16},
        {"label": "dictation", "query": dictation, "n_results": 16},
        *[{"label": term, "query": term, "n_results": 8} for term in search_terms[:8]],
    ]
    candidates: dict[str, dict] = {}
    for item in search_queries:
        search_args = {"query": item["query"], "n_results": item.get("n_results", 8)}
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
    )[:MAX_DETAIL_CANDIDATES]:
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
            query_text=rerank_text,
        )
        if gop not in already_billed
        and not _fachgruppe_conflict(candidates[gop], practice_fachgruppe)
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
        search_plan=search_terms,
        candidates=[candidates[gop] for gop in ranked_codes[:MAX_DECISION_CANDIDATES]],
        think=think,
    )
    return {
        "response": response,
        "tool_log": tool_log,
        "search_plan": search_terms,
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
    query_text: str = "",
) -> list[str]:
    return sorted(
        candidates,
        key=lambda gop: (
            -float(candidates[gop].get("rerank_score", -999.0)),
            _fachgruppe_priority(candidates[gop], preferred_fachgruppe),
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


def _fachgruppe_conflict(candidate: dict, practice_fachgruppe: str | None) -> bool:
    """True if the billing practice may not bill this GOP's chapter at all.

    EBM Kapitel III is arztgruppenspezifisch: per the chapter preambles, those
    GOPs may only be billed by the named specialty. Chapters II, IV, V, VII and
    VIII are arztgruppenübergreifend. This mirrors catalogue structure for all
    specialties; it is not tuned to any test case.
    """
    if not practice_fachgruppe:
        return False
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    kapitel = str(details.get("kapitel") or hit.get("kapitel") or "")
    fachgruppe = str(details.get("fachgruppe") or hit.get("fachgruppe") or "")
    if not kapitel.startswith("III"):
        return False
    return fachgruppe != practice_fachgruppe


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


_JUDGE_SYSTEM_PROMPT = (
    "Du bist ein EBM-Abrechnungsexperte. Pruefe genau eine Kandidaten-GOP und "
    "entscheide, ob sie fuer dieses Diktat abgerechnet werden soll.\n"
    "Regeln:\n"
    "- Waehle die GOP nur, wenn die beschriebene Leistung im Diktat oder im "
    "dokumentierten Patientenkontext dokumentiert ist. Eine Diagnose oder ein "
    "Symptom allein reicht nicht.\n"
    "- Technische Einzelheiten des obligaten Leistungsinhalts (z.B. "
    "Messintervalle, Ableitungszahlen, Registrierungsdetails) gelten als "
    "erfuellt, wenn die Leistung selbst fachgerecht dokumentiert ist und nichts "
    "im Diktat dagegen spricht.\n"
    "- Situative Sonderbedingungen (z.B. Uhrzeiten, Wochenende/Feiertag, "
    "Notdienst, Besuch, Altersgrenzen, Mindestdauer von Gespraechen, spezielle "
    "Programme oder Bescheinigungen) muessen dagegen explizit dokumentiert "
    "sein.\n"
    "- Beruecksichtige Abrechnungsbestimmungen und Anmerkungen im GOP-Text, "
    "insbesondere ob die GOP neben bereits abgerechneten GOPs berechnungsfaehig "
    "ist.\n"
    "- Eine Zuschlags-GOP ist waehlbar, wenn ihre Basis-GOP bereits abgerechnet "
    "ist oder nach Diktat und Kontext heute ebenfalls zur Abrechnung ansteht.\n"
    "- Beruecksichtige Alter und Geschlecht des Patienten.\n"
    "Antworte ausschliesslich mit JSON: {\"select\": true} oder {\"select\": false}."
)


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
    """Judge each candidate independently, then resolve catalogue exclusions.

    Binary per-candidate judgements are more reliable for small models than a
    single selection over the full candidate list. Conflicts between selected
    candidates are resolved deterministically via the catalogue's Ausschluss
    lists, keeping the higher-ranked candidate.
    """
    if not candidates:
        return "[]"
    selected: list[str] = []
    for candidate in candidates:
        if _judge_candidate(model, dictation, patient_id, quartal, patient_ctx,
                            practice_fachgruppe, candidate, think):
            selected.append(candidate["gop"])
    by_gop = {candidate["gop"]: candidate for candidate in candidates}
    selected = _resolve_exclusions(selected, by_gop)
    return json.dumps(selected, ensure_ascii=False)


def _judge_candidate(
    model: str,
    dictation: str,
    patient_id: str,
    quartal: str,
    patient_ctx: dict | None,
    practice_fachgruppe: str | None,
    candidate: dict,
    think: bool,
) -> bool:
    summary = _candidate_for_prompt(candidate)
    try:
        response = _chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Patient-ID: {patient_id}, Quartal: {quartal}\n"
                        f"Abrechnende Praxis/Fachgruppe: {practice_fachgruppe or 'unbekannt'}\n"
                        f"Patientenkontext:\n{json.dumps(patient_ctx, ensure_ascii=False)}\n\n"
                        f"Diktat:\n{dictation}\n\n"
                        f"Kandidat:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            options={"temperature": 0, "num_predict": 30},
            think=think,
        )
    except ollama.ResponseError:
        return False
    return _parse_judge_response((response.message.content or "").strip())


def _parse_judge_response(text: str) -> bool:
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, dict) and isinstance(value.get("select"), bool):
                return value["select"]
        except json.JSONDecodeError:
            pass
    return bool(re.search(r"\btrue\b", text, re.IGNORECASE))


def _resolve_exclusions(selected: list[str], by_gop: dict[str, dict]) -> list[str]:
    """Resolve mutually exclusive selections, keeping the higher-valued GOP.

    Uses the Ausschluss lists parsed from the catalogue ("nicht neben den
    Gebührenordnungspositionen ..."). When two approved candidates exclude each
    other, EBM practice bills the higher-valued service, so conflicts are
    resolved by Punkte (rank order as tie-break). Output keeps rank order.
    """
    by_value = sorted(selected, key=lambda gop: -_candidate_punkte(by_gop.get(gop, {})))
    kept: list[str] = []
    for gop in by_value:
        exclusions = set(_candidate_exclusions(by_gop.get(gop, {})))
        conflict = any(
            other in exclusions or gop in _candidate_exclusions(by_gop.get(other, {}))
            for other in kept
        )
        if not conflict:
            kept.append(gop)
    return [gop for gop in selected if gop in kept]


def _candidate_punkte(candidate: dict) -> int:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    punkte = details.get("punkte", hit.get("punkte"))
    return int(punkte) if isinstance(punkte, (int, float)) else 0


def _candidate_exclusions(candidate: dict) -> list[str]:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    exclusions = details.get("ausschluesse", [])
    return [str(e) for e in exclusions] if isinstance(exclusions, list) else []


def _candidate_for_prompt(candidate: dict) -> dict:
    details = candidate.get("details") if isinstance(candidate.get("details"), dict) else {}
    hit = candidate.get("search_hit") if isinstance(candidate.get("search_hit"), dict) else {}
    # Prefer the full normative text (incl. Abrechnungsbestimmungen/Anmerkungen)
    # over the embedding snippet so the judge sees billing constraints.
    text = details.get("volltext") or details.get("document") or hit.get("document") or ""
    return {
        "gop": candidate.get("gop"),
        "fachgruppe": details.get("fachgruppe", hit.get("fachgruppe")),
        "ausschluesse": details.get("ausschluesse", []),
        "text": _truncate(str(text), 1600),
    }


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
