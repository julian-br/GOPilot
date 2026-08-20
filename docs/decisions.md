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

**006 — Specialty codes as index metadata, filtered at query time** (2026-08-18)
648 codes share their text with a same-named code of another specialty — `03322` and `04322` have
cosine similarity 1.0, so no embedding can separate them. The specialty lists do, and they never
overlap. Codes without a list are billable by anyone, so every filter needs the specialty or an
empty specialty list (see `billable_filter`).

**007 — MLflow for experiment tracking, metrics computed ourselves** (2026-08-18)
Runs locally against a file store and records the git commit and branch by itself. LangSmith and
Langfuse were ruled out — the first is hosted and the master data licence forbids redistribution,
the second needs four services to self-host. MLflow's own retrieval metrics are deprecated in
favour of LLM-judged scoring, so recall stays a few lines of our own code: exact, deterministic
and free, which LLM-judged metrics are not.

**008 — Qdrant Local for hybrid retrieval** (2026-08-20)
Dictations quote the catalogue almost verbatim, yet dense search missed those codes entirely
(`32030`: rank 1 lexically, absent from the dense top 50). Chroma has a Search API for hybrid
retrieval, but sparse indexes are not available in the local embedded setup. Qdrant Local runs
in-process without Docker, persists on disk, and exposes LangChain-native dense, sparse and hybrid
retrieval modes. The sparse side uses FastEmbed's BM25 with German stemming, stored as a Qdrant
sparse vector with IDF weighting; dense and sparse candidates are fused with Qdrant's RRF.

**009 — Generation behind a small model client** (2026-08-20)
LLM calls are isolated behind `src.generation.client.open_chat_model`, with prompts and structured
output schemas in separate modules. The first provider is local Ollama, using `qwen3:4b`, but the
recommendation code receives a LangChain chat model so cloud providers can be swapped in later.
Initial evaluation measures the LLM without retrieval as a baseline.
