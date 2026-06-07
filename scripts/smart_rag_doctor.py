#!/usr/bin/env python3
"""smart_rag doctor — portability self-check. Run this FIRST on a new machine
(e.g. the work PC) to confirm Smart RAG will work and what mode it'll run in.

  py -3.13 scripts/smart_rag_doctor.py

Reports: Python, deps, embedder+GPU status, and whether retrieval will be
HYBRID (vectors) or KEYWORD-ONLY (graceful fallback). Smart RAG WORKS either way —
this just tells you which.
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def line(ok, label, detail=""):
    print(f"  [{'OK ' if ok else 'WARN'}] {label}" + (f" — {detail}" if detail else ""))


def main():
    print(f"smart_rag doctor — {sys.executable}")
    print(f"Python {sys.version.split()[0]}\n")

    ok_core = True
    # core deps (required)
    for mod in ("numpy", "openpyxl", "sqlite3"):
        try:
            __import__(mod); line(True, f"core dep: {mod}")
        except Exception as e:
            line(False, f"core dep MISSING: {mod}", str(e)); ok_core = False

    # optional deps (graceful) — report LIBRARY presence only (not model load)
    for mod, what in [("sentence_transformers", "embeddings library"),
                      ("torch", "GPU acceleration"),
                      ("fitz", "PDF (pymupdf)"), ("docx", "DOCX"),
                      ("tiktoken", "token counting")]:
        try:
            __import__(mod); line(True, f"optional: {what} ({mod})")
        except Exception:
            line(False, f"optional MISSING: {what} ({mod}) — feature degrades gracefully")

    # embedder — distinguish: library missing vs MODEL not cached vs working.
    print()
    try:
        import sentence_transformers  # noqa: F401
        _lib = True
    except Exception:
        _lib = False
    try:
        from smart_rag.core import embed
        if not _lib:
            line(False, "EMBEDDINGS: library not installed",
                 "→ KEYWORD-ONLY. Fix: pip install -r smart_rag/requirements.txt")
        elif embed.available():
            dev = embed._pick_device()
            line(True, "EMBEDDINGS WORK", f"device={dev} → HYBRID retrieval (best quality)")
            if dev == "cpu":
                line(True, "note", "CPU embeddings — fine for use; GPU needs cu128 torch.")
        else:
            # library IS installed but the model didn't load → almost always the
            # model just isn't downloaded yet (offline/first-run). Give the real fix.
            cached = embed._model_is_cached()
            if not cached:
                line(False, "EMBEDDINGS: model not downloaded yet (library OK)",
                     "→ KEYWORD-ONLY for now. Fix (needs internet ONCE): run  "
                     'python -c "from sentence_transformers import SentenceTransformer; '
                     "SentenceTransformer('all-MiniLM-L6-v2')\"  — then re-run doctor.")
            else:
                line(False, "EMBEDDINGS: model cached but failed to load",
                     "→ KEYWORD-ONLY. Check torch install. Smart RAG still works.")
    except Exception as e:  # noqa: BLE001
        line(False, "embed check failed", str(e))

    # smoke test: ingest + query in memory
    print()
    try:
        from smart_rag import SmartRAG
        d = SmartRAG()
        d.ingest_chunks([{"text": "The deserializer SER-100 remaps channels.",
                          "source": "t.md"}], verbose=False)
        hit = d.search_chunks("deserializer", top_k=1)
        miss = d.search_chunks("price of bitcoin", top_k=1)
        line(bool(hit) and not miss, "smoke test: query finds relevant + abstains on junk",
             f"{len(hit)} hit / {len(miss)} junk")
    except Exception as e:  # noqa: BLE001
        line(False, "smoke test failed", str(e))

    print("\n" + ("✅ READY" if ok_core else "❌ core deps missing — pip install -r smart_rag/requirements.txt"))


if __name__ == "__main__":
    main()
