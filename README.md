# Smart RAG

**Your data's first and last point. Point Smart RAG at files or logs — it distills
them into a compact, deduplicated, source-cited fact store, then answers questions
(grounded, cited) or searches (exact, no AI) with far fewer tokens and correct
answers. Works on any data shape: spreadsheets, logs, code, docs, configs, AUTOSAR.**

No cloud dependency. You own the whole path: ingestion → store → retrieval. It can
also feed its store to your own AI or RAG.

---

## Why Smart RAG (vs vector RAG and TOON)

Normal RAG chops data into thousands of embedded chunks — wasteful, lossy, and it
can't tell you when it doesn't know. TOON compresses tabular data into a compact
format — but it's *only* a format: no retrieval, no abstention, no citations.

**Smart RAG indexes _facts, not chunks._** A fact is one
`(entity, attribute, value, source, version, date)` — the real unit of truth. It
preserves structure on ingest, deduplicates, remembers provenance, retrieves the
relevant slice, abstains when it doesn't know, and cites every answer.

### Measured comparison (`python -m smart_rag.compare`)

| Data shape | Flat RAG | TOON | **Smart RAG** |
|---|---|---|---|
| **Tabular** (product spec table) | 3,219 tok/query | 72 tok (whole blob, no retrieval) | **52 tok/query, cited, abstains** |
| **Logs** (190k-line system log) | 202,705 tok | ❌ not applicable | **4,076 tok (98% smaller), deduped, abstains** |
| Retrieval | keyword, no abstention | none (send whole table) | hybrid + calibrated abstention |
| Citations | no | no | **every answer cited to source** |
| Unanswerable query | returns junk | can't abstain | **NOT_FOUND** |

**Bottom line:** TOON is a compact *format* for tabular data. Smart RAG is a
*retrieval system* that works on **any** shape, sends the least per query, abstains
honestly, and cites every answer. Run the benchmark on your own data and see.

### Smart RAG's tabular emit — TOON's good idea, improved

TOON's win is **schema-once**: declare columns once, then emit bare value rows.
Smart RAG keeps that and adds what TOON can't:

```
@schema entity, UFS:num/gb, RAM:num/gb, SXM:bool | src
SKU1001, 128, 24, true | spec.csv
SKU1002, 256, 32, false | spec.csv
```

- **Typed columns** — the unit (`gb`) is declared once in the header, so `128GB`
  becomes an unambiguous `128` (not repeated every row). Often *smaller* than TOON
  on messy data, and the LLM never re-infers types.
- **Provenance** (`| src`) — every row cites its source, so answers are groundable.
- **Retrieval-aware partial emit** — emit only the rows a query needs (Smart RAG
  already retrieved them), not the whole table. The per-query token cost is what you
  actually pay, and partial emit wins there.

See `smart_rag/core/tabular_emit.py` and `python -m smart_rag.compare`.

## Quick start

```bash
pip install -r smart_rag/requirements.txt
python scripts/smart_rag_doctor.py        # check your environment (hybrid vs keyword-only)

# CLI
python -m smart_rag.cli ingest yourfile.xlsx --db store.db
python -m smart_rag.cli ask "what is the UFS size for SKU1001" --db store.db

# Python
from smart_rag import SmartRAG
d = SmartRAG("store.db")
d.ingest("yourfolder")
print(d.answer("how does X work").to_text())   # cited, or NOT_FOUND
```

GPU is auto-detected; without the embedding model it degrades gracefully to
keyword-only retrieval (still abstains + cites). See `scripts/smart_rag_doctor.py`.

## What's inside

```
smart_rag/
  api.py          SmartRAG — ingest, answer, search, ingest_chunks
  core/
    plan.py       QueryPlan — classify a query (FACT/PROSE/RELATION/...)
    retrieve.py   hybrid retrieval + single-scale rerank + abstention
    fact.py       Fact + FactStore (entity·attribute·value, provenance)
    db.py         canonical SQLite store (source_id lifecycle, persisted vectors)
    answer.py     AnswerResult — status + evidence + confidence (the trust surface)
    relation.py   entity→entity edges (codegraph/AUTOSAR absorbed as relations)
    embed.py      self-contained embeddings (GPU→CPU, offline-capable)
  adapters/       see "Formats & standards" below
  tests/          run_all.py → adapters + lifecycle + adversarial suites
  compare.py      Smart RAG vs Flat vs TOON benchmark
  cli.py / gui.py interfaces
```

## Tested

```bash
python -m smart_rag.tests.run_all
# adapters 10/10 · lifecycle 6/6 · adversarial 12/12
```

The adversarial suite covers the cases that break naive RAG: duplicate filenames,
restart-then-update/delete, unknown-id-with-prose, schema migration, JSONL
round-trip, single-file vectors, AUTOSAR ref-ownership.

## Formats & standards

Smart RAG reads common formats **and** real engineering interchange standards —
because the entity-relation model fits structured standards naturally:

| Domain | Formats / standards |
|---|---|
| **General** | Excel/CSV/JSON, **Parquet**, **YAML/TOML**, INI/cfg/properties, Markdown/PDF/DOCX/HTML, **PPTX/Visio** |
| **Software** | source code (multi-language), **OpenAPI/Swagger** specs, codegraph symbol DBs |
| **Automotive** | **AUTOSAR ARXML**, **DBC** (CAN), **ODX** (ISO 22901 diagnostics/DTCs), **A2L** (ASAM MCD-2 MC calibration) |
| **Semiconductor** | **IP-XACT** (IEEE 1685), **SPICE netlists** (component→net graph) |
| **Logs** | DLT, logcat, slog, generic text |

Adding a format = one adapter (`smart_rag/adapters/`), 30-100 lines; the core never
changes. Adapters declare the entity types they emit (`emits`) and the `standard`
they target, so coverage is reportable ("12 DTCs, 4 services from this ODX").

Optional parsers degrade gracefully: no `pyarrow` → Parquet skipped; no `pyyaml` →
YAML skipped. Everything else keeps working.

## Design principles

- **Facts, not chunks** — dedup + structure preservation + provenance.
- **Abstain honestly** — calibrated relevance floor; unanswerable → NOT_FOUND.
- **Cite everything** — every answer maps to a source.
- **One scorer decides** — query plan + hybrid rerank (no ad-hoc routing).
- **Degrade gracefully** — no model? keyword-only, still correct.

## License

MIT — see [LICENSE](LICENSE).
