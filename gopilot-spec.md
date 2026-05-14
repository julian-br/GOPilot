# GIPILOT: Automatische Abrechnungs-Vorschläge aus Arzt-Diktaten

## Ziel

Ein lokaler Agent, der Arzt-Diktate analysiert und die korrekten EBM-GOPs (Gebührenordnungspositionen) zur Abrechnung vorschlägt, inklusive Punktzahl, Euro-Berechnung und Kompatibilitätsprüfung. Tool Calling, keine reine RAG-Pipeline.

## Hardware-Constraints

- GPU: RTX 2080 Ti (11 GB VRAM)
- Alles lokal, keine externen API-Calls

## Tech Stack

| Komponente  | Wahl                                          | Begründung                          |
| ----------- | --------------------------------------------- | ----------------------------------- |
| LLM         | Qwen 2.5 7B Instruct (Q4_K_M)                 | gutes Tool Calling, passt in VRAM   |
| LLM-Runtime | Ollama                                        | OpenAI-kompatible API, easy setup   |
| Embeddings  | nomic-embed-text via Ollama                   | klein, schnell, gut genug für Start |
| Vector-DB   | ChromaDB                                      | embedded, kein Server               |
| Sprache     | Python 3.11+                                  |                                     |
| Code-Stil   | English, keine Kommentare außer absolut nötig |                                     |

Optional später: medBERT.de als Embedder testen (besser für deutschen Medizin-Text).

## Architektur

```
Diktat (Text)
    ↓
LLM (Qwen 2.5 7B) im Tool-Calling-Loop
    ↓
Tools:
  - search_gop_by_description(query)
  - get_gop_details(gop_number)
  - check_compatibility(gop_list)
  - calculate_euro(gop_list)
    ↓
Strukturierte Ausgabe (JSON):
  - vorgeschlagene GOPs
  - Punktzahl, Euro
  - Warnungen / Ausschlüsse
```

## Datenquellen

1. **EBM-Katalog**: scrapen von `ebm.kbv.de` oder PDF-Export verwenden
   - Pro GOP: Nummer, Beschreibung, Punktzahl, Voraussetzungen, Ausschlüsse, Kapitel/Fachgruppe
2. **Orientierungspunktwert 2026**: 12,7404 Cent (hardcoded für MVP, später konfigurierbar)
3. **Test-Diktate**: synthetisch generieren mit Claude/GPT-4 API oder manuell

## Datenstruktur (ChromaDB)

Pro Chunk:

```json
{
  "id": "gop_03220",
  "content": "GOP 03220: Beratung und Behandlung durch Hausarzt, mind. 10 Min. ...",
  "metadata": {
    "gop": "03220",
    "punkte": 380,
    "kapitel": "3",
    "fachgruppe": "Hausarzt",
    "ausschluesse": ["03230"],
    "voraussetzungen": []
  }
}
```

## Tool Definitions

```python
def search_gop_by_description(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over EBM catalog. Returns GOP candidates with scores."""

def get_gop_details(gop_number: str) -> dict:
    """Exact lookup by GOP number. Returns full details."""

def check_compatibility(gop_list: list[str]) -> dict:
    """Checks mutual exclusions between GOPs.
    Returns: {compatible: bool, conflicts: list[tuple]}"""

def calculate_euro(gop_list: list[str], punktwert_cent: float = 12.7404) -> dict:
    """Returns total euro amount and per-GOP breakdown."""
```

## Implementation Steps

### Phase 1: Setup (30 min)

- [ ] Ollama installieren, Qwen 2.5 7B + nomic-embed-text pullen
- [ ] Python-Projekt setup, dependencies: `ollama`, `chromadb`, `pydantic`, `httpx`
- [ ] Hello-World Tool Calling Test mit Ollama

### Phase 2: EBM-Daten (1-2h)

- [ ] EBM-Daten beschaffen (scrape ebm.kbv.de oder PDF parsen)
- [ ] Pro GOP in ChromaDB indexieren mit Metadaten
- [ ] Sanity-Check: 5-10 manuelle Queries testen

### Phase 3: Tools bauen (1h)

- [ ] Die 4 Tools als reine Python-Funktionen implementieren
- [ ] Unit-Tests mit bekannten Inputs
- [ ] OpenAI-Tool-Schema Definitionen

### Phase 4: Agent Loop (1h)

- [ ] System-Prompt schreiben (mit Few-Shot Beispiel für Tool-Use)
- [ ] Tool-Calling-Loop implementieren (max. 10 Iterationen, dann abbrechen)
- [ ] JSON-Output validieren mit Pydantic

### Phase 5: Test-Diktate + Eval (1h)

- [ ] 20-30 synthetische Diktate generieren mit Ground-Truth-GOPs
- [ ] Evaluation-Script: Precision/Recall pro Diktat
- [ ] Confusion-Analyse: welche GOPs werden verwechselt?

## Test-Diktate Beispiel

```
Diktat:
"Patient Müller, 67, kam zur Routineuntersuchung. Habe ihn zur
Hypertonie beraten, etwa 15 Minuten. Blutdruck gemessen: 145/90.
Ein EKG haben wir auch gemacht, war unauffällig. Folgerezept für
Ramipril ausgestellt."

Erwartete GOPs:
- 03220 (Beratung Hausarzt, ≥10min)
- (Blutdruckmessung ist meist in 03220 enthalten, nicht separat)
- 03321 (EKG Hausarzt) -- falls Voraussetzungen erfüllt
- ggf. Verordnungs-Position
```

## Evaluation-Metriken

Pro Test-Diktat:

- **Precision**: Von vorgeschlagenen GOPs - wie viele sind korrekt?
- **Recall**: Von korrekten GOPs - wie viele wurden gefunden?
- **F1** über Test-Set
- **Euro-Differenz**: vorgeschlagener Betrag vs. korrekter Betrag

## Stretch Goals

1. **Whisper-Integration**: echtes Audio → Diktat-Text → GOPs
2. **Fachgruppen-Filter**: Hausarzt vs. Facharzt vs. Psychotherapeut
3. **Quartal-Logik**: GOPs, die nur 1×/Quartal erlaubt sind
4. **UI**: kleines Gradio/Streamlit-Frontend
5. **Re-Ranking**: BM25 + Semantic hybrid für bessere Retrieval-Qualität
6. **Constrained Decoding**: erzwinge JSON-Output via grammar / outlines

## Wichtige Notizen / Fallstricke

- **Tool Calling bei 7B-Modellen ist wackelig**: bei schlechter Performance Few-Shot-Beispiele im System-Prompt nutzen
- **Chunking**: pro GOP ein Chunk - nicht naiv nach Token-Anzahl chunken, sonst wird der Zusammenhang zerrissen
- **Lizenz EBM-Daten**: für persönliches Lernprojekt unkritisch, für Veröffentlichung klären
- **Medizinische Verantwortung**: System ist Vorschlag, kein Ersatz für ärztliche Prüfung. Disclaimer im Output.
- **VRAM-Budget**: Qwen 2.5 7B (~5 GB) + Embedder (~500 MB) = ~6 GB. Lässt 5 GB für Context-Window. Reicht für 4-8K Token Context.

## Definition of Done (MVP)

- Bei einem realistischen Test-Diktat schlägt das System ≥80% der korrekten GOPs vor
- Falsche GOPs werden mit Precision ≥70% vermieden
- Euro-Berechnung ist immer korrekt (deterministisch, nicht vom LLM gemacht)
- Ende-zu-Ende-Zeit pro Diktat: <30 Sekunden auf 2080 Ti

## Repo-Struktur

```
ebm-bot/
├── data/
│   ├── ebm_raw/           # gescrapte/heruntergeladene Rohdaten
│   └── test_dictations/   # synthetische Test-Diktate + Labels
├── src/
│   ├── ingest.py          # EBM-Daten in ChromaDB laden
│   ├── tools.py           # die 4 Tools
│   ├── agent.py           # Tool-Calling-Loop
│   ├── schemas.py         # Pydantic-Modelle
│   └── eval.py            # Evaluation-Script
├── notebooks/
│   └── exploration.ipynb  # für interaktives Testen
├── pyproject.toml
└── README.md
```
