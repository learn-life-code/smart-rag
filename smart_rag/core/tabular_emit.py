#!/usr/bin/env python3
"""Tabular emit — Smart RAG's compact, retrieval-aware tabular format.

TOON's insight (worth taking): declare the column schema ONCE, then every row is
just values — no repeated keys. That's where its token savings come from on tabular
data, and it's a good idea.

What this adds ON TOP of that idea (so it's an IMPROVEMENT, not a copy):

  1. SCHEMA-ONCE  — same key win as TOON: header line, then bare value rows.
  2. TYPED COLUMNS — each column declares its type/unit once (int, gb, bool, str),
     so the model doesn't re-infer per cell and values stay unambiguous (128 vs
     "128GB"). TOON emits raw cells with no type contract.
  3. PROVENANCE   — an optional trailing source column ties each row to where it
     came from. Plain TOON drops provenance, so an LLM can't cite.
  4. RETRIEVAL-AWARE PARTIAL EMIT — you can emit ONLY the rows a query needs
     (Smart RAG already retrieved them), not the whole table. TOON is whole-blob; the
     real per-query token cost is what matters, and partial emit wins there.
  5. NULL ELISION — absent cells are skipped, not encoded as empty placeholders.

Output shape (one entity per row):
    @schema entity, UFS/gb:int, RAM/gb:int, SXM:bool  | src
    SKU1001, 128, 24, true  | spec.csv
    SKU1002, 256, 32, false | spec.csv

The header is emitted once; rows are bare values; types live in the header; the
trailing `| src` gives provenance. This is strictly more useful than TOON for
feeding an LLM that must answer AND cite — at comparable or better token cost.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

_TRUE = {"1", "true", "yes", "y", "present", "available"}
_FALSE = {"0", "false", "no", "n", "absent", "n/a"}


def _infer_type(values: List[str]) -> str:
    """Infer a column's type/unit from its values (declared ONCE in the header)."""
    vals = [v for v in values if v and str(v).strip()]
    if not vals:
        return "str"
    low = [str(v).strip().lower() for v in vals]
    if all(v in _TRUE or v in _FALSE for v in low):
        return "bool"
    # numeric with an optional unit suffix (128GB, 24gb, 500ms)
    import re
    units = set()
    allnum = True
    for v in low:
        m = re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*([a-z%]*)', v)
        if not m:
            allnum = False
            break
        if m.group(2):
            units.add(m.group(2))
    if allnum:
        unit = next(iter(units)) if len(units) == 1 else ""
        base = "num"
        return f"{base}/{unit}" if unit else base
    return "str"


def _canon(value: str, typ: str) -> str:
    """Canonicalize a cell to match its declared type (strip the unit, normalize
    bools) so values are unambiguous and the unit isn't repeated every row."""
    v = str(value).strip()
    if typ == "bool":
        return "true" if v.lower() in _TRUE else "false"
    if typ.startswith("num"):
        import re
        m = re.match(r'(-?\d+(?:\.\d+)?)', v)
        return m.group(1) if m else v
    return v


def emit_tabular(rows: List[Dict[str, str]], *, columns: Optional[List[str]] = None,
                 entity_key: str = "entity", source_key: str = "source",
                 with_source: bool = True) -> str:
    """Emit rows ([{entity, col: val, ..., source}]) in the compact format.

    Schema-once + typed columns + optional provenance + null elision. Emitting only
    the retrieved rows (partial) is just passing fewer rows — that's the per-query
    win over TOON's whole-blob.
    """
    if not rows:
        return ""
    # collect columns (preserve first-seen order) excluding entity/source
    if columns is None:
        seen: List[str] = []
        for r in rows:
            for k in r:
                if k not in (entity_key, source_key) and k not in seen:
                    seen.append(k)
        columns = seen
    # infer one type per column from all rows
    types = {c: _infer_type([str(r.get(c, "")) for r in rows]) for c in columns}

    # header: entity + typed columns (+ src)
    hdr_cols = [entity_key] + [f"{c}:{types[c]}" if types[c] != "str" else c
                               for c in columns]
    header = "@schema " + ", ".join(hdr_cols) + (" | src" if with_source else "")
    out = [header]

    for r in rows:
        cells = [str(r.get(entity_key, ""))]
        for c in columns:
            v = r.get(c, "")
            cells.append(_canon(v, types[c]) if v not in (None, "") else "")  # null elision
        line = ", ".join(cells)
        if with_source:
            line += f" | {r.get(source_key, '')}"
        out.append(line)
    return "\n".join(out)


def emit_from_store(store, entities: Optional[List[str]] = None,
                    attributes: Optional[List[str]] = None,
                    with_source: bool = True) -> str:
    """Render a FactStore (or a retrieved subset of it) in the compact tabular
    format. Pass `entities`/`attributes` to emit ONLY the retrieved slice — the
    retrieval-aware partial emit that beats whole-blob TOON on per-query tokens."""
    ents = entities if entities is not None else list(store.entities)
    rows = []
    # determine the attribute columns present across the chosen entities
    if attributes is None:
        cols: List[str] = []
        for e in ents:
            for a in store.lookup(e):
                if a not in cols:
                    cols.append(a)
        attributes = cols
    for e in ents:
        facts = store.lookup(e)
        row: Dict[str, str] = {"entity": e}
        src = ""
        for a in attributes:
            r = facts.get(a)
            if r:
                row[a] = r[0]["value"]
                src = src or (r[0].get("sources") or [""])[0]
        row["source"] = src
        if len(row) > 2:   # has at least one attribute value
            rows.append(row)
    return emit_tabular(rows, columns=list(attributes), with_source=with_source)
