"""
Data engineering pipeline: fetch latest KBV EBM PDF and ingest into ChromaDB.

URL pattern (KBV):
  current quarter : https://www.kbv.de/documents/praxis/abrechnung/ebm/{year}-{q}-ebm.pdf
  archived quarter: https://www.kbv.de/documents/praxis/abrechnung/ebm/archiv/{year}-{q}-ebm.pdf

Usage:
    python -m src.fetch_ebm          # fetch latest + ingest
    python -m src.fetch_ebm --fetch-only   # only download PDF
    python -m src.fetch_ebm --ingest-only  # only ingest already-downloaded PDF
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import httpx

from src.ingest import ingest

BASE_URL = "https://www.kbv.de/documents/praxis/abrechnung/ebm"
RAW_DIR = Path("data/ebm_raw")
TIMEOUT = 30


def _current_quarter() -> tuple[int, int]:
    today = date.today()
    return today.year, (today.month - 1) // 3 + 1


def _candidate_urls(year: int, quarter: int) -> list[tuple[str, str]]:
    """Return (url, filename) candidates to try, newest first."""
    candidates = []
    y, q = year, quarter
    for _ in range(8):  # look back up to 8 quarters
        filename = f"ebm_{y}_q{q}.pdf"
        # current quarter lives at /ebm/, older ones at /ebm/archiv/
        candidates.append((f"{BASE_URL}/{y}-{q}-ebm.pdf", filename))
        candidates.append((f"{BASE_URL}/archiv/{y}-{q}-ebm.pdf", filename))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return candidates


def fetch_latest_pdf() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    year, quarter = _current_quarter()

    for url, filename in _candidate_urls(year, quarter):
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"Already downloaded: {dest}")
            return dest

        print(f"Trying {url} ...", end=" ", flush=True)
        try:
            with httpx.stream("GET", url, timeout=TIMEOUT, follow_redirects=True) as r:
                if r.status_code != 200:
                    print(f"HTTP {r.status_code}")
                    continue
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            print(f"OK -> {dest}")
            return dest
        except httpx.RequestError as e:
            print(f"Error: {e}")
            continue

    print("ERROR: Could not find a downloadable EBM PDF.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch + ingest latest KBV EBM PDF")
    parser.add_argument("--fetch-only", action="store_true", help="Only download, skip ingest")
    parser.add_argument("--ingest-only", action="store_true", help="Skip download, ingest existing PDF")
    args = parser.parse_args()

    if args.ingest_only:
        pdfs = sorted(RAW_DIR.glob("ebm_*.pdf"))
        if not pdfs:
            print(f"No PDF found in {RAW_DIR}", file=sys.stderr)
            sys.exit(1)
        pdf_path = pdfs[-1]  # newest by name
        print(f"Using existing PDF: {pdf_path}")
    else:
        pdf_path = fetch_latest_pdf()

    if not args.fetch_only:
        ingest(pdf_path)


if __name__ == "__main__":
    main()
