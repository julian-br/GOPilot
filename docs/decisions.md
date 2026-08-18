# Decisions

Major architectural choices, newest last.

**001 — LangChain as the framework** (2026-08-18)
Replaces hand-wired Ollama calls. Only `langchain-core` until a step needs more.

**002 — EBM from the KBV master data feed, not the PDF** (2026-08-18)
PDF parsing lost entry boundaries and detached prices from codes; the master data has the rules
as fields. Licensed for use, not redistribution — stays out of version control.

**003 — Embed `langtext` + `leistungsinhalt_obligat` only** (2026-08-18)
The optional content is generic boilerplate; rules stay metadata. Finds procedures, not flat-rate
codes — those need a second mechanism.

**004 — Index the whole catalogue, filter by specialty at query time** (2026-08-18)
A GP bills far beyond chapter 03.

**005 — Five fields to start, rule attributes deferred** (2026-08-18)
They must return: the conditions are nowhere in the prose.

**006 — Specialty codes as index metadata, filtered with `$contains`** (2026-08-18)
648 codes share their text with a same-named code of another specialty — `03322` and `04322` have
cosine similarity 1.0, so no embedding can separate them. The specialty lists do, and they never
overlap. Codes without a list are billable by anyone and carry `["*"]`, so every filter needs the
`$or` (see `billable_by`).

**007 — MLflow for experiment tracking, metrics computed ourselves** (2026-08-18)
Runs locally against a file store and records the git commit and branch by itself. LangSmith and
Langfuse were ruled out — the first is hosted and the master data licence forbids redistribution,
the second needs four services to self-host. MLflow's own retrieval metrics are deprecated in
favour of LLM-judged scoring, so recall stays a few lines of our own code: exact, deterministic
and free, which LLM-judged metrics are not.
