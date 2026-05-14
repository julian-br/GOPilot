"""
Embedding quality test: checks whether expected GOPs appear in ChromaDB top-N results.

ChromaDB with cosine space returns distances where distance = 1 - cosine_similarity.
Similarity of 1.0 = identical, 0.0 = orthogonal, negative = opposite.

Usage:
    python -m tests.test_embeddings
    python -m tests.test_embeddings --top 10 --verbose
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("SSLKEYLOGFILE", None)

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingest import CHROMA_PATH, COLLECTION_NAME, OllamaEmbedder

# Each probe: (query, expected_gop, label)
# Queries represent what the agent would search for, not full dictations.
PROBES = [
    # Direct procedure names
    ("Belastungs-EKG Elektrokardiographie Hausarzt",          "03321", "EKG Hausarzt"),
    ("Langzeit-Blutdruckmessung ABDM 24h",                    "03324", "ABDM Hausarzt"),
    ("Spirometrie Lungenfunktion FEV1 exspiratorisch",        "03330", "Spirometrie"),
    ("Infusion intravenös i.v. Kurzinfusion",                 "02100", "IV Infusion"),
    ("Schnelltest Antigennachweisverfahren Streptokokken Gruppe A Kind", "32152", "Strep-Schnelltest Kind"),
    ("Verwaltungskomplex Überweisung Einweisung Rezept",      "01430", "Verwaltungskomplex"),
    # Pauschalen — fachgruppe now in document
    ("Versichertenpauschale Hausarzt allgemeinmedizinisch",   "03000", "Versichertenpauschale HA"),
    ("Versichertenpauschale unvorhergesehene Inanspruchnahme","03030", "Versichertenpauschale Notdienst"),
    ("Versichertenpauschale Kinder Jugendmedizin",            "04000", "Versichertenpauschale Kinder"),
    # Semantic / indirect
    ("Ergometrie Belastungstest EKG Hausarzt",                "03321", "Belastungs-EKG (semantisch)"),
    ("Spirographie Atemfluss Atemvolumen Lungenkapazität",    "03330", "Spirometrie (synonym)"),
    ("Notfallpauschale organisierter Notfalldienst Not(-fall)dienst", "01210", "Notfallpauschale"),
    ("Labortest Schnelltest Untersuchung visuell Reaktion",   "32030", "Schnelltest (generisch)"),
]

# Colloquial probes: raw dictation snippets — tests retrieval from natural language directly.
# Expected: lower recall than terminological probes. The agent must NOT feed raw dictation to
# ChromaDB; it should extract procedure terms first (LLM step), then search.
# These probes confirm what DOESN'T work so we know where the agent needs to translate.
# Source: data/test_dictations/case_*.json
COLLOQUIAL_PROBES = [
    # case_001: EKG abgeleitet → 03321
    (
        "EKG abgeleitet, Sinusrhythmus, keine Auffälligkeiten",
        "03321",
        "Diktat: EKG abgeleitet",
    ),
    # case_001: Überweisung ausgestellt → 01430
    (
        "Überweisung zum Diabetologen ausgestellt",
        "01430",
        "Diktat: Überweisung ausgestellt",
    ),
    # case_002: Langzeit-RR → 03324
    (
        "Langzeit-Blutdruckmessung angelegt, Patient gibt Gerät morgen ab",
        "03324",
        "Diktat: Langzeit-RR angelegt",
    ),
    # case_003: Strep-Test positiv → 32030 (adult patient, general rapid test)
    (
        "Schnelltest auf Streptokokken A durchgeführt: positiv",
        "32030",
        "Diktat: Strep-Test positiv",
    ),
    # case_006: Infusion → 02100
    (
        "Infusion Furosemid 40mg i.v. angelegt, Patient stabilisiert",
        "02100",
        "Diktat: Furosemid i.v.",
    ),
    # case_006: Spirometrie → 03330
    (
        "Lungenfunktion gemessen: FEV1/FVC 0,61, Hinweis auf obstruktive Ventilationsstörung",
        "03330",
        "Diktat: Lungenfunktion FEV1/FVC",
    ),
    # case_006: stationäre Einweisung → 01430
    (
        "Stationäre Einweisung in Kardiologie ausgestellt",
        "01430",
        "Diktat: Stationäre Einweisung",
    ),
]


def cosine_similarity(distance: float) -> float:
    """ChromaDB cosine distance = 1 - cosine_similarity."""
    return round(1.0 - distance, 4)


def _query_probes(
    col, probes: list[tuple], top_k: int
) -> list[tuple]:
    results = []
    for query, expected, label in probes:
        res = col.query(query_texts=[query], n_results=top_k, include=["metadatas", "distances"])
        gops = [m["gop"] for m in res["metadatas"][0]]
        sims = [cosine_similarity(d) for d in res["distances"][0]]
        rank = next((i + 1 for i, g in enumerate(gops) if g == expected), None)
        expected_sim = sims[rank - 1] if rank else None
        results.append((label, expected, rank, gops, sims, rank is not None, expected_sim, sims[0]))
    return results


def _print_section(results: list[tuple], top_k: int, title: str) -> int:
    n = len(results)
    hits = {1: 0, 3: 0, top_k: 0}
    for label, expected, rank, gops, _, found, expected_sim, top1_sim in results:
        if found:
            for k in hits:
                if rank <= k:
                    hits[k] += 1

    print(f"\n{title}  (top-{top_k}, n={n})")
    print(f"  Hit@1 : {hits[1]:2d}/{n}  ({hits[1]/n*100:.0f}%)")
    print(f"  Hit@3 : {hits[3]:2d}/{n}  ({hits[3]/n*100:.0f}%)")
    print(f"  Hit@{top_k:<2}: {hits[top_k]:2d}/{n}  ({hits[top_k]/n*100:.0f}%)")
    hit_sims = [r[6] for r in results if r[6] is not None]
    if hit_sims:
        print(f"  Avg sim of correct hits: {sum(hit_sims)/len(hit_sims):.4f}")
    print()
    for label, expected, rank, gops, _, found, expected_sim, top1_sim in results:
        if found:
            marker = "OK" if rank == 1 else "~"
            sim_str = f"sim={expected_sim:.4f}"
            status = f"rank {rank}  {sim_str}" if rank == 1 else f"rank {rank}  {sim_str}  (top1={top1_sim:.4f})"
        else:
            marker, status = "X", f"MISS  top1={gops[0]} sim={top1_sim:.4f}"
        print(f"  [{marker}] {label:<47}  exp={expected}  {status}")
    return sum(1 for r in results if not r[5])  # missed count


def run(top_k: int = 5, verbose: bool = False) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=COLLECTION_NAME, embedding_function=OllamaEmbedder())

    term_results = _query_probes(col, PROBES, top_k)
    coll_results = _query_probes(col, COLLOQUIAL_PROBES, top_k)

    missed_term = _print_section(term_results, top_k, "Terminologische Probes (präzise Fachbegriffe)")
    # Colloquial probes are informational — lower recall is expected (agent must extract terms first)
    _print_section(coll_results, top_k, "Umgangssprachliche Probes (Arztdiktate, informativ)")

    if verbose:
        print("\n--- Detailed results ---")
        for r in term_results + coll_results:
            label, expected, gops, sims = r[0], r[1], r[3], r[4]
            print(f"\n{label} (expected: {expected})")
            for i, (g, s) in enumerate(zip(gops, sims)):
                mark = " <--" if g == expected else ""
                print(f"  {i+1}. {g}  sim={s:.4f}{mark}")

    if missed_term:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(top_k=args.top, verbose=args.verbose)


if __name__ == "__main__":
    main()
