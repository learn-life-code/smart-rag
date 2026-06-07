#!/usr/bin/env python3
"""Codegraph adapter — read a codegraph.db (nodes + edges) into Smart RAG.

This is the COORDINATION the user asked about: Smart RAG does NOT replace codegraph.
Codegraph keeps its specialized binary symbol + call-graph extraction (ELF/dex/.so
parsers). This adapter ABSORBS its output so Smart RAG's hybrid retrieval + abstention
+ citations apply to code structure too — one query interface over content AND
code-graph.

  nodes  → symbol facts: entity=symbol, attribute=kind|defined_in|signature
  edges  → RELATION facts: entity=src, attribute=edge_kind (calls/imports/contains/
           instantiates/inherits), value=tgt  (queryable via "what calls X")

So you build BOTH indexes of a software folder (codegraph for structure, Smart RAG for
content); Smart RAG then answers structural questions using codegraph's edges with v2
quality, and content questions from its own adapters.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

# codegraph edge kinds → Smart RAG relation attributes (whole-word, lowercased)
_EDGE_ATTR = {"calls": "calls", "imports": "imports", "contains": "contains",
              "instantiates": "instantiates", "inherits": "inherits",
              "implements": "implements", "references": "references"}


class CodegraphAdapter(Adapter):
    # A codegraph.db (sqlite) with nodes+edges. Matched by name, not just suffix.
    suffixes = ()
    name = "codegraph"

    def can_handle(self, path: str) -> bool:
        b = os.path.basename(path).lower()
        if not b.endswith(".db"):
            return False
        # confirm it has the codegraph schema (nodes + edges)
        try:
            c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tabs = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            c.close()
            return "nodes" in tabs and "edges" in tabs
        except Exception:
            return False

    def extract(self, path: str) -> Iterable[Fact]:
        src = f"{os.path.basename(path)} (codegraph)"
        try:
            c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except Exception:
            return
        try:
            cols = {r[1] for r in c.execute("PRAGMA table_info(nodes)")}
            name_col = "name" if "name" in cols else ("qualified_name" if "qualified_name" in cols else None)
            if not name_col:
                return
            # node id → symbol name (for resolving edges)
            id_to_name = {}
            sel = f"id, {name_col}" + (", kind" if "kind" in cols else "") + \
                  (", file_path" if "file_path" in cols else "") + \
                  (", signature" if "signature" in cols else "")
            for row in c.execute(f"SELECT {sel} FROM nodes"):
                nid, nm = row[0], row[1]
                if not nm:
                    continue
                id_to_name[nid] = nm
                i = 2
                if "kind" in cols and len(row) > i and row[i]:
                    yield Fact(entity=nm, attribute="kind", value=str(row[i]),
                               source=src, kind="extracted"); i += 1
                if "file_path" in cols and len(row) > i and row[i]:
                    yield Fact(entity=nm, attribute="defined_in", value=str(row[i]),
                               source=src, kind="extracted"); i += 1
                if "signature" in cols and len(row) > i and row[i]:
                    yield Fact(entity=nm, attribute="signature", value=str(row[i])[:200],
                               source=src, kind="extracted"); i += 1
            # edges → relation facts (src --kind--> tgt)
            ecols = {r[1] for r in c.execute("PRAGMA table_info(edges)")}
            if {"source", "target", "kind"} <= ecols:
                for s, t, k in c.execute("SELECT source, target, kind FROM edges"):
                    sn, tn = id_to_name.get(s), id_to_name.get(t)
                    if not sn or not tn:
                        continue
                    attr = _EDGE_ATTR.get(str(k).lower(), str(k).lower())
                    yield Fact(entity=sn, attribute=attr, value=tn,
                               source=src, kind="relation")
        finally:
            c.close()

    def prose_chunks(self, path: str):
        return []
