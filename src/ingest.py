"""
Pipeline: KBV EBM PDF -> parsed GOP entries -> ChromaDB with Ollama embeddings.

Usage:
    python -m src.ingest --pdf data/ebm_raw/ebm_2026_q2.pdf
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import ollama
import pdfplumber
from chromadb import Documents, EmbeddingFunction, Embeddings

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "ebm_gops"
EMBED_MODEL = "qwen3-embedding:4b"
BATCH_SIZE = 50

# Real GOP header lines always end with a euro amount, e.g.:
# "01210 Notfallpauschale I ... 15,29 €"
_GOP_LINE = re.compile(r"^(\d{5})\s+(.+\d+,\d+\s*€\s*)$")

# Fallback for GOPs without euro amount (Kostenpauschalen, Pseudo-GOPs)
_GOP_LINE_NO_EURO = re.compile(r"^(\d{5})\s+(.+)")
_DESCRIPTION_HAS_WORDS = re.compile(r"[A-Za-zÀ-ɏ]{4,}")
# Filter out sentence continuations that wrap onto a new line starting with a GOP number
_DESCRIPTION_IS_CONTINUATION = re.compile(
    r"^(bis |und [\d,]|,\s*\d|ist |oder |nicht |f.r |bei |in |an |von |zu |ab |auf Basis |"
    r"berechnungsf|Kontakt |Arzt-|gem\.|gem |nach |sowie |auf die )"
)
_DESCRIPTION_MIN_LEN = 15  # "Versichertenpauschale" = 20 chars must pass

# Points: "120 Punkte" — number directly before "Punkte"
_PUNKTE = re.compile(r"(\d+)\s+Punkte")

# Euro amount on GOP header line: "15,29 €" (used to trim description)
_EURO = re.compile(r"\s+\d+,\d+\s*€.*$")

# Exclusions in body text — PDF garbles umlauts, so match with dot wildcard:
# "nicht neben den Geb?hrenordnungspositionen 01100, 01102 ..."
_AUSSCHLUSS_BLOCK = re.compile(
    r"nicht neben den Geb.hrenordnungsposition\w*\s+([\d][\d,\s]+)",
    re.IGNORECASE,
)
_GOP_NUMBER = re.compile(r"\b(\d{5})\b")

# Fachgruppe derived from GOP number prefix — more reliable than PDF header parsing.
_GOP_FACHGRUPPE: dict[str, tuple[str, str]] = {
    "01": ("II",     "Arztgruppenübergreifende allgemeine GOPs"),
    "02": ("II",     "Arztgruppenübergreifende allgemeine GOPs"),
    "03": ("III.a",  "Hausärztlicher Versorgungsbereich"),
    "04": ("III.b",  "Kinder- und Jugendmedizin"),
    "05": ("III.c",  "Anästhesiologie"),
    "06": ("III.d",  "Augenheilkunde"),
    "07": ("III.e",  "Chirurgie"),
    "08": ("III.f",  "Frauenheilkunde und Geburtshilfe"),
    "09": ("III.g",  "HNO"),
    "10": ("III.h",  "Hautärzte"),
    "11": ("III.i",  "Humangenetik"),
    "13": ("III.k",  "Innere Medizin und Kardiologie"),
    "14": ("III.l",  "Kinder- und Jugendpsychiatrie"),
    "15": ("III.m",  "Mund-Kiefer-Gesichtschirurgie"),
    "16": ("III.n",  "Neurologie und Neurochirurgie"),
    "17": ("III.o",  "Nuklearmedizin"),
    "18": ("III.p",  "Orthopädie"),
    "19": ("III.q",  "Pathologie"),
    "20": ("III.r",  "Sprach-, Stimm- und kindliche Hörstörungen"),
    "21": ("III.s",  "Psychiatrie und Psychotherapie"),
    "22": ("III.t",  "Psychosomatische Medizin"),
    "23": ("III.u",  "Psychotherapie"),
    "24": ("III.v",  "Radiologie"),
    "25": ("III.w",  "Strahlentherapie"),
    "26": ("III.x",  "Urologie"),
    "27": ("III.y",  "Physikalische und Rehabilitative Medizin"),
    "30": ("IV",     "Spezielle Versorgungsbereiche"),
    "31": ("IV",     "Ambulante Operationen und Anästhesien"),
    "32": ("IV",     "Laboratoriumsmedizin"),
    "33": ("IV",     "Ultraschalldiagnostik"),
    "34": ("IV",     "Radiologie und Computertomographie"),
    "35": ("IV",     "Psychotherapie-Richtlinie"),
    "36": ("IV",     "Belegärztliche Operationen"),
    "37": ("IV",     "Versorgungsverträge"),
    "38": ("IV",     "Delegationsleistungen"),
    "40": ("V",      "Kostenpauschalen"),
    "50": ("VII",    "Ambulante spezialfachärztliche Versorgung"),
    "51": ("VII",    "Ambulante spezialfachärztliche Versorgung"),
    "61": ("VIII",   "Erprobungsverfahren"),
}


# Body lines that mark the start of billing metadata — stop enriching embedding here.
# "Obligater/Fakultativer Leistungsinhalt" headers are kept; lines under them
# (prefixed with "- ") contain the actual service content and are useful for embeddings.
_BODY_STOP = re.compile(
    r"^(Abrechnungsbestimmung|Anmerkung|Berichtspflicht|"
    r"Die Geb.hrenordnung|Aufwand|Kalkulationszeit|Pr.fzeit|Stand )",
    re.IGNORECASE,
)


@dataclass
class GOPEntry:
    gop: str
    description: str
    punkte: int = 0
    kapitel: str = ""
    fachgruppe: str = ""
    ausschluesse: list[str] = field(default_factory=list)
    body: str = ""  # first meaningful content lines, used to enrich embedding
    primary_match: bool = False  # True if matched by euro-anchored regex (more reliable)

    def to_document(self) -> str:
        parts = [f"GOP {self.gop}: {self.description}."]
        if self.body:
            parts.append(self.body)
        if self.fachgruppe:
            parts.append(f"Fachgruppe: {self.fachgruppe}.")
        if self.kapitel:
            parts.append(f"Kapitel: {self.kapitel}.")
        if self.punkte:
            parts.append(f"Punkte: {self.punkte}.")
        if self.ausschluesse:
            parts.append(f"Ausschluesse: {', '.join(self.ausschluesse[:10])}.")
        return " ".join(parts)

    def to_metadata(self) -> dict:
        return {
            "gop": self.gop,
            "punkte": self.punkte,
            "kapitel": self.kapitel,
            "fachgruppe": self.fachgruppe,
            "ausschluesse": json.dumps(self.ausschluesse),
        }


class OllamaEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        response = ollama.embed(model=EMBED_MODEL, input=list(input))
        return response.embeddings


def _clean_description(raw: str) -> str:
    """Strip trailing euro amount from the GOP header line."""
    return _EURO.sub("", raw).strip()


def _extract_exclusions(block_text: str) -> list[str]:
    """Extract all 5-digit GOP numbers from an exclusion sentence."""
    exclusions = []
    for m in _AUSSCHLUSS_BLOCK.finditer(block_text):
        nums = _GOP_NUMBER.findall(m.group(1))
        exclusions.extend(nums)
    return list(dict.fromkeys(exclusions))  # deduplicate, preserve order


def parse_pdf(pdf_path: Path) -> list[GOPEntry]:
    entries: list[GOPEntry] = []
    current: GOPEntry | None = None
    current_body: list[str] = []
    current_body_content: list[str] = []  # lines before _BODY_STOP, for body enrichment
    body_stopped = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                gop_m = _GOP_LINE.match(stripped)
                is_primary = gop_m is not None
                if not gop_m:
                    fb = _GOP_LINE_NO_EURO.match(stripped)
                    desc = fb.group(2) if fb else ""
                    if (fb
                            and len(desc) >= _DESCRIPTION_MIN_LEN
                            and _DESCRIPTION_HAS_WORDS.search(desc)
                            and not _DESCRIPTION_IS_CONTINUATION.match(desc)):
                        gop_m = fb
                if gop_m:
                    # Save previous entry
                    if current is not None:
                        body = "\n".join(current_body)
                        if current.punkte == 0:
                            pm = _PUNKTE.search(body)
                            if pm:
                                current.punkte = int(pm.group(1))
                        current.ausschluesse = _extract_exclusions(body)
                        current.body = " ".join(current_body_content[:8])[:400]
                        entries.append(current)

                    gop_code = gop_m.group(1)
                    kapitel, fachgruppe = _GOP_FACHGRUPPE.get(gop_code[:2], ("", ""))
                    current = GOPEntry(
                        gop=gop_code,
                        description=_clean_description(gop_m.group(2)),
                        kapitel=kapitel,
                        fachgruppe=fachgruppe,
                        primary_match=is_primary,
                    )
                    current_body = []
                    current_body_content = []
                    body_stopped = False
                    continue

                if current is None:
                    continue

                current_body.append(stripped)

                if not body_stopped:
                    if _BODY_STOP.match(stripped):
                        body_stopped = True
                    else:
                        current_body_content.append(stripped)

                # Punkte often on line immediately after GOP header
                if current.punkte == 0:
                    pm = _PUNKTE.search(stripped)
                    if pm:
                        current.punkte = int(pm.group(1))

    # Flush last entry
    if current is not None:
        body = "\n".join(current_body)
        if current.punkte == 0:
            pm = _PUNKTE.search(body)
            if pm:
                current.punkte = int(pm.group(1))
        current.ausschluesse = _extract_exclusions(body)
        current.body = " ".join(current_body_content[:8])[:400]
        entries.append(current)

    # Deduplicate: prefer primary (euro-anchored) matches; among those, highest punkte,
    # then longest description. Fallback matches often capture cross-references.
    best: dict[str, GOPEntry] = {}
    for e in entries:
        prev = best.get(e.gop)
        if prev is None:
            best[e.gop] = e
        elif e.primary_match and not prev.primary_match:
            best[e.gop] = e  # primary always beats fallback
        elif not e.primary_match and prev.primary_match:
            pass  # keep prev
        elif e.punkte > prev.punkte:
            best[e.gop] = e
        elif e.punkte == prev.punkte and len(e.description) > len(prev.description):
            best[e.gop] = e

    return list(best.values())


def ingest(pdf_path: Path) -> None:
    print(f"Parsing {pdf_path.name}...")
    entries = parse_pdf(pdf_path)
    if not entries:
        print("ERROR: No GOP entries found. Check PDF format.", file=sys.stderr)
        sys.exit(1)
    print(f"  Parsed {len(entries)} GOP entries")

    # Sanity check: show first 3
    for e in entries[:3]:
        print(f"  Sample: GOP {e.gop} | {e.description[:60]} | {e.punkte} Punkte")

    print(f"Connecting to ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and storing (model: {EMBED_MODEL}, batch size: {BATCH_SIZE})...")
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[f"gop_{e.gop}" for e in batch],
            documents=[e.to_document() for e in batch],
            metadatas=[e.to_metadata() for e in batch],
        )
        done = min(i + BATCH_SIZE, len(entries))
        print(f"  {done}/{len(entries)}", end="\r", flush=True)

    print(f"\nDone. {len(entries)} GOPs stored in ChromaDB.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EBM PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to KBV EBM PDF")
    args = parser.parse_args()
    if not args.pdf.exists():
        print(f"ERROR: {args.pdf} not found", file=sys.stderr)
        sys.exit(1)
    ingest(args.pdf)


if __name__ == "__main__":
    main()
