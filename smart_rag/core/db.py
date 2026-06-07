#!/usr/bin/env python3
"""Canonical SQLite store for Smart RAG v2 — correct source lifecycle + persistence.

Fixes the data-integrity blockers from the review:
  * STABLE SOURCE IDENTITY: every source has a source_id = hash(abspath). Facts,
    prose and embeddings carry it. Replace/delete is BY source_id (never basename),
    so two same-named files in different folders never collide.
  * ATOMIC REPLACE: a file's new rows are written in ONE transaction that first
    deletes its old rows by source_id; on parse failure the previous version stays.
  * HONEST COVERAGE: per-source status (ok|empty|error|unsupported) + counts.
  * PERSISTED EMBEDDINGS: prose vectors stored as BLOBs → no re-embed on restart;
    only changed sources are re-embedded.
  * SCHEMA VERSION + MIGRATIONS + BACKUP/RESTORE.
  * FTS5/BM25 helpers for hybrid retrieval.

SQLite is canonical. JSONL is kept as a lossless EXPORT only (until the v2 gate
passes, then dropped).
"""
from __future__ import annotations

import os
import sqlite3
import struct
import time
from typing import Dict, List, Optional

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS sources(
  source_id TEXT PRIMARY KEY,   -- hash(abspath), stable identity
  path TEXT,                    -- abspath (display); NOT used as identity
  hash TEXT,                    -- content sha1 (incremental: skip if unchanged)
  parser TEXT, parser_version TEXT,
  status TEXT,                  -- ok | empty | error | unsupported
  error TEXT, n_facts INTEGER, n_prose INTEGER, ts REAL);

CREATE TABLE IF NOT EXISTS facts(
  source_id TEXT, entity TEXT, attribute TEXT, value TEXT, source TEXT,
  version TEXT, date TEXT, confidence REAL, span TEXT, location TEXT, kind TEXT);

CREATE TABLE IF NOT EXISTS prose(
  rowid INTEGER PRIMARY KEY,
  source_id TEXT, text TEXT, title TEXT, source TEXT, version TEXT);

-- persisted embeddings: one BLOB per prose row (float32 little-endian)
CREATE TABLE IF NOT EXISTS prose_vectors(
  prose_rowid INTEGER PRIMARY KEY, vec BLOB);
"""

# Indexes are created AFTER migration, so they never reference a column that a
# pre-v2 table is still missing (that crashed schema setup with "no such column").
_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_facts_sid ON facts(source_id);
CREATE INDEX IF NOT EXISTS ix_facts_entity ON facts(entity);
CREATE INDEX IF NOT EXISTS ix_facts_attr ON facts(attribute);
CREATE INDEX IF NOT EXISTS ix_prose_sid ON prose(source_id);
"""

_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  entity, attribute, value, content='facts', content_rowid='rowid');
CREATE VIRTUAL TABLE IF NOT EXISTS prose_fts USING fts5(
  text, title, content='prose', content_rowid='rowid');
"""


def content_hash(path: str, _b: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_b)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def _pack(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))


def _unpack(buf: bytes):
    import numpy as np
    return np.frombuffer(buf, dtype="<f4")


class DistillDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)   # tables only (no indexes yet)
        self._migrate()                    # ALTER old tables to add missing columns
        self.conn.executescript(_INDEXES)  # now safe — columns exist
        self.fts = self._try_fts()

    def _try_fts(self) -> bool:
        try:
            self.conn.executescript(_FTS)
            return True
        except sqlite3.OperationalError:
            return False

    # ── schema version + migrations ──────────────────────────────────────────
    def _migrate(self) -> None:
        cur = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        ver = int(cur[0]) if cur else 0
        if ver == SCHEMA_VERSION:
            return
        # ACTUALLY upgrade older schemas (don't just bump the version, which left
        # pre-v2 DBs missing source_id/location/kind → "no such column" crashes).
        # Add any columns the live schema expects but an old table lacks.
        def _cols(tbl):
            try:
                return {r[1] for r in self.conn.execute(f"PRAGMA table_info({tbl})")}
            except sqlite3.OperationalError:
                return set()
        _add = {
            "facts": {"source_id": "TEXT", "location": "TEXT", "kind": "TEXT"},
            "prose": {"source_id": "TEXT"},
            "sources": {"source_id": "TEXT", "parser_version": "TEXT",
                        "status": "TEXT", "n_facts": "INTEGER", "n_prose": "INTEGER"},
        }
        for tbl, cols in _add.items():
            have = _cols(tbl)
            if not have:
                continue   # table will be created fresh by the schema script
            for col, typ in cols.items():
                if col not in have:
                    try:
                        self.conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
                    except sqlite3.OperationalError:
                        pass
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES('schema_version',?)",
                          (str(SCHEMA_VERSION),))
        self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    # ── incremental support ──────────────────────────────────────────────────
    def source_unchanged(self, source_id: str, h: str) -> bool:
        r = self.conn.execute(
            "SELECT hash, status FROM sources WHERE source_id=?", (source_id,)).fetchone()
        return bool(r and r[0] == h and r[1] in ("ok", "empty"))

    def known_source_ids(self) -> set:
        return {r[0] for r in self.conn.execute("SELECT source_id FROM sources")}

    def source_path(self, source_id: str) -> Optional[str]:
        r = self.conn.execute("SELECT path FROM sources WHERE source_id=?",
                              (source_id,)).fetchone()
        return r[0] if r else None

    # ── ATOMIC replace by source_id ──────────────────────────────────────────
    def replace_source(self, source_id: str, path: str, h: str, parser: str,
                       facts: List, prose: List[dict], *, error: str = "") -> str:
        """Atomically replace one source's data. Returns status (ok|empty|error).
        On a non-empty error, the PREVIOUS version is preserved (no delete)."""
        status = "error" if error else ("empty" if not facts and not prose else "ok")
        c = self.conn
        try:
            c.execute("BEGIN")
            if status != "error":
                # delete old rows for THIS source only, by stable id
                c.execute("DELETE FROM prose_vectors WHERE prose_rowid IN "
                          "(SELECT rowid FROM prose WHERE source_id=?)", (source_id,))
                c.execute("DELETE FROM facts WHERE source_id=?", (source_id,))
                c.execute("DELETE FROM prose WHERE source_id=?", (source_id,))
                if facts:
                    c.executemany(
                        "INSERT INTO facts(source_id,entity,attribute,value,source,"
                        "version,date,confidence,span,location,kind) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        [(source_id, f.entity, f.attribute, f.value, f.source, f.version,
                          f.date, f.confidence, f.span, f.location, f.kind) for f in facts])
                if prose:
                    c.executemany(
                        "INSERT INTO prose(source_id,text,title,source,version) "
                        "VALUES(?,?,?,?,?)",
                        [(source_id, p.get("text", ""), p.get("title", ""),
                          p.get("source", ""), p.get("version", "")) for p in prose])
            c.execute(
                "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?,?)",
                (source_id, path, h, parser, "1", status, error,
                 len(facts), len(prose), time.time()))
            c.execute("COMMIT")
        except Exception as e:  # noqa: BLE001
            c.execute("ROLLBACK")
            raise
        return status

    def remove_source(self, source_id: str) -> None:
        c = self.conn
        c.execute("BEGIN")
        c.execute("DELETE FROM prose_vectors WHERE prose_rowid IN "
                  "(SELECT rowid FROM prose WHERE source_id=?)", (source_id,))
        c.execute("DELETE FROM facts WHERE source_id=?", (source_id,))
        c.execute("DELETE FROM prose WHERE source_id=?", (source_id,))
        c.execute("DELETE FROM sources WHERE source_id=?", (source_id,))
        c.execute("COMMIT")

    def rebuild_fts(self) -> None:
        if not self.fts:
            return
        self.conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        self.conn.execute("INSERT INTO prose_fts(prose_fts) VALUES('rebuild')")
        self.conn.commit()

    # ── persisted embeddings ─────────────────────────────────────────────────
    def prose_needing_vectors(self):
        """rows (rowid, text, title) of prose with NO stored vector yet."""
        return self.conn.execute(
            "SELECT p.rowid, p.text, p.title FROM prose p "
            "LEFT JOIN prose_vectors v ON v.prose_rowid=p.rowid "
            "WHERE v.prose_rowid IS NULL").fetchall()

    def store_vectors(self, rowid_vecs) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO prose_vectors VALUES(?,?)",
                              [(rid, _pack(v)) for rid, v in rowid_vecs])
        self.conn.commit()

    def load_vectors(self):
        """Return (matrix[n,dim], [prose_row...]) for the whole corpus, or (None,[])."""
        import numpy as np
        rows = self.conn.execute(
            "SELECT p.rowid, p.text, p.title, p.source, p.version, v.vec "
            "FROM prose p JOIN prose_vectors v ON v.prose_rowid=p.rowid").fetchall()
        if not rows:
            return None, []
        mats = [_unpack(r[5]) for r in rows]
        chunks = [{"text": r[1], "title": r[2], "source": r[3], "version": r[4]}
                  for r in rows]
        return np.vstack(mats).astype("float32"), chunks

    # ── load hot layer ───────────────────────────────────────────────────────
    def load_into(self, store) -> None:
        # NAMED arguments — the SELECT column order (…location,kind,source_id) does
        # NOT match the Fact field order (…source_id,location,kind). Positional
        # Fact(*row) swapped source_id↔kind, corrupting restart-then-update/delete
        # (remove_source_id couldn't match). Bind by name to be order-independent.
        from smart_rag.core.fact import Fact
        for (entity, attribute, value, source, version, date, confidence, span,
             location, kind, source_id) in self.conn.execute(
                "SELECT entity,attribute,value,source,version,date,confidence,span,"
                "location,kind,source_id FROM facts"):
            store.add(Fact(entity=entity, attribute=attribute, value=value,
                           source=source, version=version, date=date,
                           confidence=confidence, span=span, location=location,
                           kind=kind, source_id=source_id))
        for row in self.conn.execute(
                "SELECT text,title,source,version,source_id FROM prose"):
            store.add_prose(row[0], row[2], row[3], row[1], source_id=row[4])

    # ── FTS5 / BM25 ──────────────────────────────────────────────────────────
    def fts_facts(self, query: str, limit: int = 30):
        if not self.fts:
            return []
        q = " OR ".join(w for w in _terms(query))
        if not q:
            return []
        try:
            return self.conn.execute(
                "SELECT entity,attribute,value,source FROM facts WHERE rowid IN "
                "(SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY rank "
                "LIMIT ?)", (q, limit)).fetchall()
        except sqlite3.OperationalError:
            return []

    def fts_prose(self, query: str, limit: int = 20):
        if not self.fts:
            return []
        q = " OR ".join(w for w in _terms(query))
        if not q:
            return []
        try:
            return self.conn.execute(
                "SELECT text,title,source FROM prose WHERE rowid IN "
                "(SELECT rowid FROM prose_fts WHERE prose_fts MATCH ? ORDER BY rank "
                "LIMIT ?)", (q, limit)).fetchall()
        except sqlite3.OperationalError:
            return []

    # ── coverage / health ────────────────────────────────────────────────────
    def coverage(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*), SUM(n_facts), SUM(n_prose) FROM sources "
            "GROUP BY status").fetchall()
        by = {r[0]: {"files": r[1], "facts": r[2] or 0, "prose": r[3] or 0} for r in rows}
        tot = self.conn.execute("SELECT COUNT(*),SUM(n_facts),SUM(n_prose) FROM sources").fetchone()
        return {"by_status": by, "sources": tot[0] or 0,
                "facts": tot[1] or 0, "prose": tot[2] or 0,
                "schema_version": self.get_meta("schema_version")}

    def stats(self) -> dict:
        c = self.conn
        return {"sources": c.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "facts": c.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                "prose": c.execute("SELECT COUNT(*) FROM prose").fetchone()[0],
                "vectors": c.execute("SELECT COUNT(*) FROM prose_vectors").fetchone()[0]}

    # ── backup / restore ─────────────────────────────────────────────────────
    def backup(self, dest: str) -> None:
        self.conn.commit()
        bk = sqlite3.connect(dest)
        with bk:
            self.conn.backup(bk)
        bk.close()

    def integrity_ok(self) -> bool:
        r = self.conn.execute("PRAGMA integrity_check").fetchone()
        return bool(r and r[0] == "ok")

    def close(self) -> None:
        try:
            self.conn.commit(); self.conn.close()
        except Exception:
            pass


def _terms(query: str):
    import re
    return [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
