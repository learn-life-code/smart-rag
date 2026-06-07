#!/usr/bin/env python3
"""IndexManager — named, persistent Smart RAG indexes for agent workflows.

Point a NAME at a source once; query it forever, fast and cited. Indexes live in
~/.smartrag/indexes/<name>.db (SQLite, persisted embeddings) so re-opening is
instant and refreshes are incremental (only changed files re-ingested).

This is what makes Smart RAG an agent's search reflex: instead of grep+read across
a workspace (thousands of tokens), the agent calls answer(name, query) and gets
cited facts in ~50 tokens.

    mgr = IndexManager()
    mgr.build("repo", "/path/to/repo")          # once (or to refresh)
    mgr.answer("repo", "how does auth work")     # AnswerResult: cited, abstains
    mgr.search("repo", entity="UserService")     # exact facts, no LLM
    mgr.list()                                   # what's indexed
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from smart_rag.api import SmartRAG
from smart_rag.collectors.fs import collect_fs


def _home() -> Path:
    base = Path(os.environ.get("SMARTRAG_HOME", str(Path.home() / ".smartrag")))
    (base / "indexes").mkdir(parents=True, exist_ok=True)
    return base


class IndexManager:
    """Manage named, persistent indexes under ~/.smartrag/."""

    def __init__(self, home: Optional[str] = None):
        self.home = Path(home) if home else _home()
        (self.home / "indexes").mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.home / "catalog.json"
        self._open: dict = {}   # name -> SmartRAG (lazy)

    # ── catalog ──────────────────────────────────────────────────────────────
    def _catalog(self) -> dict:
        if self.catalog_path.exists():
            try:
                return json.loads(self.catalog_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_catalog(self, cat: dict) -> None:
        self.catalog_path.write_text(json.dumps(cat, indent=2))

    def _db_path(self, name: str) -> str:
        return str(self.home / "indexes" / f"{name}.db")

    # ── build / refresh ──────────────────────────────────────────────────────
    def build(self, name: str, source: str, *, max_file_mb: float = 25,
              verbose: bool = True) -> dict:
        """Index (or refresh) `source` under `name`. Incremental: unchanged files
        are skipped, so re-running after edits is fast. Returns stats."""
        db = self._db_path(name)
        sr = SmartRAG(db)
        files = list(collect_fs(source, max_file_mb=max_file_mb))
        if verbose:
            print(f"[smartrag] indexing '{name}': {len(files)} files from {source}")
        used = empty = skipped = errs = 0
        t0 = time.time()
        from smart_rag.adapters import adapter_for
        for i, f in enumerate(files, 1):
            if not adapter_for(f):
                skipped += 1
                continue
            st = sr.ingest(f, verbose=False)
            used += st.get("files_ingested", 0)
            empty += st.get("files_empty", 0)
            skipped += st.get("files_skipped_unchanged", 0)
            errs += len(st.get("errors", []))
            if verbose and i % 200 == 0:
                print(f"[smartrag]   {i}/{len(files)} files…")
        stats = sr.store.stats() if hasattr(sr.store, "stats") else {}
        self._open[name] = sr
        cat = self._catalog()
        cat[name] = {"source": os.path.abspath(source), "db": db,
                     "files_seen": len(files), "ingested": used,
                     "entities": stats.get("entities"),
                     "facts": stats.get("distinct_facts"),
                     "updated": time.time(), "build_secs": round(time.time() - t0, 1)}
        self._save_catalog(cat)
        if verbose:
            print(f"[smartrag] '{name}' ready: {used} ingested, {empty} empty, "
                  f"{errs} errored in {cat[name]['build_secs']}s")
        return cat[name]

    def build_ssh(self, name: str, target: str, *, key: Optional[str] = None,
                  password: Optional[str] = None, port: int = 22,
                  extra_commands: Optional[dict] = None, verbose: bool = True) -> dict:
        """Index a LIVE machine over SSH (read-only). Same named-index workflow as
        build(), but the source is a host's discovery output instead of a folder."""
        from smart_rag.collectors.ssh import collect_ssh_chunks
        if verbose:
            print(f"[smartrag] ssh-collecting '{name}' from {target} (read-only)…")
        chunks = collect_ssh_chunks(target, key=key, password=password, port=port,
                                    extra_commands=extra_commands)
        sr = SmartRAG(self._db_path(name))
        sr.ingest_chunks(chunks, verbose=False)
        self._open[name] = sr
        stats = sr.store.stats() if hasattr(sr.store, "stats") else {}
        cat = self._catalog()
        cat[name] = {"source": f"ssh:{target}", "db": self._db_path(name),
                     "chunks": len(chunks), "entities": stats.get("entities"),
                     "facts": stats.get("distinct_facts"), "updated": time.time()}
        self._save_catalog(cat)
        if verbose:
            print(f"[smartrag] '{name}' ready: {len(chunks)} command blocks indexed.")
        return cat[name]

    # ── query ────────────────────────────────────────────────────────────────
    def _get(self, name: str) -> Optional[SmartRAG]:
        if name in self._open:
            return self._open[name]
        db = self._db_path(name)
        if not os.path.exists(db):
            return None
        sr = SmartRAG(db)          # reopens with persisted facts + vectors (instant)
        self._open[name] = sr
        return sr

    def answer(self, name: str, query: str):
        """Cited, abstaining answer (AnswerResult). The agent's fast path."""
        sr = self._get(name)
        if sr is None:
            from smart_rag.core.answer import AnswerResult
            return AnswerResult(status="NOT_FOUND", query=query, confidence="LOW",
                                answer=f"No index named '{name}'. Build it first.")
        return sr.answer(query)

    def search(self, name: str, **kwargs):
        """Exact fact lookup (no LLM): entity=, attribute=, value_contains=."""
        sr = self._get(name)
        return sr.search(**kwargs) if sr else []

    def ask_text(self, name: str, query: str) -> str:
        return self.answer(name, query).to_text()

    # ── live verification (snapshot → current truth) ─────────────────────────
    def verify(self, name: str, query: str, *, key: Optional[str] = None,
               password: Optional[str] = None) -> dict:
        """Answer from the index (fast snapshot), THEN re-run the source command
        LIVE on the SSH target to confirm it's still true. Returns
        {answer, snapshot, live, command, changed}. Only for ssh-collected indexes.

        This keeps the boundary honest: the index gives a fast cited POINTER; verify
        checks it against the device NOW (read-only). If the live output differs from
        the snapshot, `changed` is True — the fast answer is stale, trust the live one.
        """
        cat = self._catalog().get(name, {})
        source = cat.get("source", "")
        if not source.startswith("ssh:"):
            return {"error": f"'{name}' is not an SSH index (source={source}); "
                    "live verification only applies to SSH-collected indexes."}
        target = source[4:]
        res = self.answer(name, query)
        # the command is embedded in the matched evidence text as a '$ <cmd>' line
        sr = self._get(name)
        cmd = ""
        snap = ""
        import re as _re
        if res.evidence:
            snap = res.evidence[0].text
            m = _re.search(r'^\$ (.+)$', snap, _re.M)
            if not m:
                # fall back: find the full prose chunk this evidence came from
                top = res.evidence[0].source
                for p in (sr.store.prose if sr else []):
                    if top in p.get("title", "") or top in p.get("text", "")[:100]:
                        snap = p.get("text", "")
                        m = _re.search(r'^\$ (.+)$', snap, _re.M)
                        break
            if m:
                cmd = m.group(1).strip()
        if not cmd:
            return {"answer": res.to_text(), "command": "",
                    "note": "No source command found to re-run; verify manually."}
        from smart_rag.collectors.ssh import run_one
        try:
            live = run_one(target, cmd, key=key, password=password)
        except Exception as e:  # noqa: BLE001
            return {"answer": res.to_text(), "command": cmd,
                    "error": f"live re-run failed: {e}"}
        # crude change detection: compare the meaningful body lines
        def _body(t):
            return "\n".join(ln for ln in t.splitlines()
                             if ln and not ln.startswith(("#", "$ ", "##")))
        changed = _body(snap).strip() != live.strip()
        return {"answer": res.to_text(), "command": cmd, "snapshot": _body(snap)[:1500],
                "live": live[:1500], "changed": changed}

    # ── admin ────────────────────────────────────────────────────────────────
    def list(self) -> dict:
        return self._catalog()

    def remove(self, name: str) -> bool:
        cat = self._catalog()
        if name not in cat:
            return False
        try:
            for ext in ("", "-wal", "-shm"):
                p = Path(self._db_path(name) + ext)
                if p.exists():
                    p.unlink()
        except OSError:
            pass
        self._open.pop(name, None)
        del cat[name]
        self._save_catalog(cat)
        return True
