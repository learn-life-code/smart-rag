# Smart RAG — your data's first and last point

**Point Smart RAG at your files or logs. It distills them into a compact,
deduplicated, source-cited fact store. Then ask questions (grounded answers) or
search (exact, no AI) — with far fewer tokens, a smaller index, and correct,
cited answers.**

No external RAG, no cloud dependency, you own the whole path: ingestion → store →
retrieval. It can also feed its store to an AI or another RAG if you want.

---

## Why it's different

Normal RAG chops your data into thousands of text chunks and embeds them. On real
data that's wasteful and lossy:
- Versioned/duplicated files → the same content indexed dozens of times.
- Flat chunking destroys structure (merged headers, multi-row tables) → wrong answers.
- A "list everything" query retrieves so many chunks it overflows the model and
  silently drops most of your data.

Smart RAG indexes **facts, not chunks**. A *fact* is one
`(entity, attribute, value, source, version, date)` — the real unit of truth. It
preserves structure on the way in, deduplicates, and remembers where every value
came from and which version it changed in.

## Proven results (on real engineering data)

| | Product spec table (Excel) | A 190k-line log |
|---|---|---|
| Index size | structure recovered, 97 facts/part | **190,434 lines → 682 facts (279× smaller)** |
| Tokens to answer one query | 35,829 → **346 (99% less)** | 1,195,489 → **840 (~1400× less)** |
| Lookup latency | 0.21 ms | 0.12 ms |
| Correctness | **4/4**, incl. a PCB part number a flat RAG got **wrong** | components→errors, deduped |

## Try it (5 minutes, on YOUR data)

```bash
# CLI
py -3.13 -m smart_rag.cli ingest  yourfile.xlsx --save store.jsonl
py -3.13 -m smart_rag.cli ask     "what is the <attr> for <id>"  yourfile.xlsx
py -3.13 -m smart_rag.cli search  --entity <id> --attr <attr>    yourfile.xlsx
py -3.13 -m smart_rag.cli profile yourfile.xlsx        # what shape is my data?
py -3.13 -m smart_rag.bench       yourfile.xlsx        # show me the numbers

# GUI (drag-a-file, no coding)
py -3.13 distill/gui.py
```

```python
# Library
from smart_rag import SmartRAG
d = SmartRAG()
d.ingest("yourfolder")                       # file or folder, mixed formats
print(d.ask("UFS and RAM for SKU1001"))   # grounded + sources
d.search(entity="SKU1001", attribute="ufs")   # programmatic, no LLM
```

## Format support (honest — ✅ = proven & tested)

| Format | Status | Notes |
|---|---|---|
| Excel .xlsx/.xlsm | ✅ | structure-preserving (merged cells, multi-row headers) → facts |
| CSV / JSON | ✅ | tabular / list-of-records → facts |
| Logs .log/.txt/.dlt/.slog | ✅ | component → errors/events over time, deduped |
| Docs .md / .rst | ✅ | header-aware sections → semantic prose ("how does X work") |
| PDF | ✅ | datasheets, specs, reports → prose (needs `pymupdf`) |
| Word .docx | ✅ | ODX, specs, reports → prose (needs `python-docx`) |
| Code .py/.sh/.ps1/.c/.cpp/.h/.js/.ts/.go/.rs | ✅ | symbols → facts + function+comment → prose |
| AUTOSAR .arxml / CAN .dbc / .xml | ✅ | signals/messages → facts |
| PPT / Visio / MS-Project / Jira | 🔜 | as demand shows |

**Install everything:** `py -3.13 -m pip install -r smart_rag/requirements.txt`
(Semantic search + PDF/Word need the optional libs; without them, those formats are
skipped and text/tabular still work.)

## Extending to a new format

Add one adapter — the core never changes:

```python
from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

class MyAdapter(Adapter):
    suffixes = (".myext",)
    def extract(self, path):
        yield Fact(entity="...", attribute="...", value="...", source=path)
```

Register it in `distill/adapters/__init__.py`. Done.

## The grounding guarantee

Smart RAG never invents a value. `ask` answers only from distilled facts and cites
the source; if a value isn't there, it says so. (An optional LLM only *phrases*
the answer — the facts are the source of truth.)
