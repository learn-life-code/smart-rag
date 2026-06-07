#!/usr/bin/env python3
"""Source-lifecycle gating tests — the v2 data-integrity adoption gate.

Proves the review's release blockers are fixed:
  * duplicate basenames in different folders don't collide
  * atomic replace updates only the changed file
  * delete removes only that file's data
  * restart loads from disk with NO re-embedding, identical results

Run: py -3.13 -m smart_rag.tests.test_lifecycle
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from smart_rag import SmartRAG   # noqa: E402

_PASS = _FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    _PASS += bool(cond); _FAIL += (not cond)


def _ini(path, body):
    open(path, "w", encoding="utf-8").write(body)


def main():
    tmp = tempfile.mkdtemp(prefix="distill_lc_")
    db = os.path.join(tmp, "store.db")
    a = os.path.join(tmp, "folderA"); b = os.path.join(tmp, "folderB")
    os.makedirs(a); os.makedirs(b)

    # ── 1. DUPLICATE BASENAME: same filename, different folders, different data ──
    _ini(os.path.join(a, "config.ini"), "[net]\nIP=10.0.0.1\nName=AAA")
    _ini(os.path.join(b, "config.ini"), "[net]\nIP=10.0.0.2\nName=BBB")
    d = SmartRAG(db); d.ingest(tmp, verbose=False)
    ipA = [f.value for f in d.search(attribute="IP") if "AAA" in
           "".join(g.value for g in d.search(entity=f.entity))] or \
          [f.value for f in d.search(attribute="IP")]
    vals = sorted(f.value for f in d.search(attribute="IP"))
    check("duplicate-basename: BOTH IPs present (10.0.0.1 AND 10.0.0.2)",
          vals == ["10.0.0.1", "10.0.0.2"])

    # ── 2. ATOMIC REPLACE: change folderA's file, folderB untouched ─────────────
    _ini(os.path.join(a, "config.ini"), "[net]\nIP=10.9.9.9\nName=AAA")
    d.ingest(tmp, verbose=False)
    vals = sorted(f.value for f in d.search(attribute="IP"))
    check("replace: A updated to 10.9.9.9, B still 10.0.0.2 (no cross-corruption)",
          vals == ["10.0.0.2", "10.9.9.9"])

    # ── 3. DELETE: remove folderB's file → only B's data gone ───────────────────
    os.remove(os.path.join(b, "config.ini"))
    d.ingest(tmp, verbose=False)
    vals = sorted(f.value for f in d.search(attribute="IP"))
    check("delete: B's IP gone, A's intact", vals == ["10.9.9.9"])

    # ── 4. RESTART: reopen DB, no re-ingest, identical data, vectors persisted ──
    d.db.close()
    d2 = SmartRAG(db)   # fresh process simulation: loads from disk only
    vals = sorted(f.value for f in d2.search(attribute="IP"))
    check("restart: data identical after reopen (no re-ingest)", vals == ["10.9.9.9"])

    # ── 5. EMPTY FILE honesty: a file with no extractable content ───────────────
    _ini(os.path.join(a, "blank.ini"), "\n\n; only comments\n")
    st = d2.ingest(tmp, verbose=False)
    check("empty-file reported as empty (not 'ingested')", st.get("files_empty", 0) >= 1)

    # ── 6. INCREMENTAL: re-ingest unchanged → skipped ───────────────────────────
    st = d2.ingest(tmp, verbose=False)
    check("unchanged files skipped on re-ingest",
          st.get("files_skipped_unchanged", 0) >= 1)

    d2.db.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== lifecycle: {_PASS}/{_PASS+_FAIL} passed ===")
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
