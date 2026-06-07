#!/usr/bin/env python3
"""The universal unit of Smart RAG: a Fact, and the FactStore that holds them.

Everything ingested — a spreadsheet cell, a log line, a contract clause — becomes
a Fact: a single (entity, attribute, value) observation with provenance and time.
This is what lets one engine serve any format: adapters differ, Facts don't.

A Fact is deliberately small and flat so the store stays compact and fast:

    Fact(entity, attribute, value, source, version, date, confidence, span)

The FactStore deduplicates identical observations, tracks every source/version a
value was seen in (so it can cite provenance and surface version conflicts), and
answers entity/attribute queries by sub-millisecond lookup — never inventing.
"""
from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple


def source_id_for(path: str) -> str:
    """Stable identity for a file = hash of its ABSOLUTE path. Two files with the
    same basename in different folders get DIFFERENT ids → no cross-source
    collision on replace/delete (the v2 data-integrity fix)."""
    try:
        ap = os.path.abspath(path)
    except Exception:
        ap = path
    return hashlib.sha1(ap.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass(frozen=True)
class Fact:
    entity: str            # the thing the fact is about (part no, component, party)
    attribute: str         # which property (UFS Size, error_code, payment_term)
    value: str             # the value (128, EADDRNOTAVAIL, "Net 30")
    source: str = ""       # human-readable origin (file::sheet, log:line, doc:page)
    version: str = ""      # version/build/run the observation belongs to
    date: str = ""         # ISO date if the observation is time-stamped
    confidence: float = 1.0  # 1.0 = read verbatim; <1 = inferred/extracted
    span: str = ""         # optional raw span for audit ("...UFS256S...")
    # Stable identity of the FILE this came from (hash of abspath). Used for
    # replace/delete so two same-named files in different folders never collide.
    source_id: str = ""
    location: str = ""     # precise locator within the source (cell/line/page)
    kind: str = "extracted"  # "extracted" (verbatim) | "inferred" (derived)


class FactStore:
    """Entity → attribute → list of Fact observations, with fast grounded lookup."""

    def __init__(self) -> None:
        self._facts: Dict[str, Dict[str, List[Fact]]] = defaultdict(lambda: defaultdict(list))
        self.n_observations = 0
        # Prose chunks kept alongside facts so one ingest serves narrative queries too.
        self.prose: List[dict] = []   # [{text, source, version}]
        self._attr_cache: Optional[List[str]] = None
        self.attr_index = None        # built lazily for attribute-intent resolution
        self.prose_index = None       # built lazily for semantic prose search

    # ── write ────────────────────────────────────────────────────────────────
    # Truly-absent markers — these mean "no value", NOT a real zero/false. Only
    # these are dropped. A literal 0 / false / "no" is a REAL value (e.g. SXM=0
    # means "no SXM", which is different from the attribute being missing). The
    # earlier version dropped "0" — a correctness bug for boolean/count/bit fields.
    _NULL_VALUES = frozenset({"", "none", "nan", "null", "n/a", "na", "#n/a",
                              "-", "--", "—"})

    def add(self, fact: Fact) -> None:
        if not fact.entity or not fact.attribute:
            return
        v = "" if fact.value is None else str(fact.value).strip()
        if v.lower() in self._NULL_VALUES:
            return
        self._facts[fact.entity][fact.attribute].append(fact)
        self.n_observations += 1
        self._attr_cache = None       # invalidate attribute vocabulary cache

    def add_many(self, facts: Iterable[Fact]) -> None:
        for f in facts:
            self.add(f)

    def remove_source_id(self, source_id: str) -> None:
        """Drop all facts + prose from one file (by stable id) — for atomic replace
        / source deletion in the hot layer, mirroring the DB. NEVER by basename."""
        if not source_id:
            return
        for ent in list(self._facts.keys()):
            for attr in list(self._facts[ent].keys()):
                kept = [f for f in self._facts[ent][attr] if f.source_id != source_id]
                if kept:
                    self._facts[ent][attr] = kept
                else:
                    del self._facts[ent][attr]
            if not self._facts[ent]:
                del self._facts[ent]
        self.prose = [p for p in self.prose if p.get("source_id") != source_id]
        self._attr_cache = None
        self.prose_index = None

    def add_prose(self, text: str, source: str = "", version: str = "",
                  title: str = "", source_id: str = "") -> None:
        if text and text.strip():
            self.prose.append({"text": text, "source": source, "version": version,
                               "title": title, "source_id": source_id})
            self.prose_index = None   # invalidate semantic index

    # ── stats ────────────────────────────────────────────────────────────────
    @property
    def entities(self) -> List[str]:
        return list(self._facts.keys())

    def distinct_fact_count(self) -> int:
        return sum(len({f.value for f in facts})
                   for e in self._facts for facts in self._facts[e].values())

    def distinct_attributes(self) -> List[str]:
        """The (small, finite) attribute vocabulary — the key to intent resolution."""
        if self._attr_cache is None:
            attrs = set()
            for e in self._facts.values():
                attrs.update(e.keys())
            self._attr_cache = sorted(attrs)
        return self._attr_cache

    def ensure_attr_index(self):
        """Build (once) the embedded attribute index used for intent resolution."""
        if self.attr_index is None:
            from smart_rag.core.intent import AttributeIndex
            self.attr_index = AttributeIndex(self.distinct_attributes())
        return self.attr_index

    def ensure_prose_index(self):
        """Build (once) the embedded prose index for semantic 'how it works' search."""
        if self.prose_index is None and self.prose:
            from smart_rag.core.prose_index import ProseIndex
            self.prose_index = ProseIndex(self.prose)
        return self.prose_index

    def stats(self) -> dict:
        return {
            "entities": len(self._facts),
            "observations": self.n_observations,
            "distinct_facts": self.distinct_fact_count(),
            "prose_chunks": len(self.prose),
        }

    # ── read / lookup ────────────────────────────────────────────────────────
    def lookup(self, entity: str, attr_hint: Optional[str] = None) -> Dict[str, List[dict]]:
        """Return {attribute: [{value, versions, sources, latest_date, latest}]}.

        Distinct values per attribute, each carrying provenance and the versions it
        appeared in. Multiple values = a version conflict the caller can surface.
        """
        e = self._facts.get(entity)
        if not e:
            # case-insensitive / suffix-tolerant fallback (SKU1001-B1 → 751…)
            e = self._fuzzy_entity(entity)
            if not e:
                return {}
        out: Dict[str, List[dict]] = {}
        for attr, obs in e.items():
            if attr_hint and attr_hint.lower() not in attr.lower():
                continue
            by_val: Dict[str, dict] = {}
            for f in obs:
                d = by_val.setdefault(f.value, {"value": f.value, "versions": set(),
                                                "sources": set(), "dates": set()})
                if f.version:
                    d["versions"].add(f.version)
                if f.source:
                    d["sources"].add(f.source)
                if f.date:
                    d["dates"].add(f.date)
            rows = [{
                "value": d["value"],
                "versions": sorted(d["versions"]),
                "sources": sorted(d["sources"]),
                "latest_date": max(d["dates"]) if d["dates"] else "",
            } for d in by_val.values()]
            rows.sort(key=lambda r: r["latest_date"], reverse=True)
            if rows:
                rows[0]["latest"] = True
            out[attr] = rows
        return out

    def _fuzzy_entity(self, entity: str) -> Optional[Dict[str, List[Fact]]]:
        el = entity.lower().strip()
        for k, v in self._facts.items():
            if k.lower() == el or k.lower().startswith(el) or el.startswith(k.lower()):
                return v
        return None

    def search(self, *, entity: Optional[str] = None, attribute: Optional[str] = None,
               value_contains: Optional[str] = None, limit: int = 50) -> List[Fact]:
        """Programmatic / keyword retrieval over facts — NO LLM needed.

        Any combination of filters; returns matching Fact observations. This is the
        'last point of retrieval' surface for keyword/automation use.
        """
        out: List[Fact] = []
        if entity is not None:
            # a SPECIFIC entity was requested — if it isn't present, return NOTHING
            # (never fall back to all entities, which returned unrelated facts).
            if entity not in self._facts:
                return []
            ents = [entity]
        else:
            ents = list(self._facts.keys())
        for ent in ents:
            for attr, obs in self._facts[ent].items():
                if attribute and attribute.lower() not in attr.lower():
                    continue
                for f in obs:
                    if value_contains and value_contains.lower() not in f.value.lower():
                        continue
                    out.append(f)
                    if len(out) >= limit:
                        return out
        return out

    # ── persistence (jsonl, compact) — LOSSLESS: facts AND prose ──────────────
    def to_jsonl(self) -> str:
        import json
        lines = []
        for ent in self._facts:
            for attr, obs in self._facts[ent].items():
                for f in obs:
                    lines.append(json.dumps(asdict(f), ensure_ascii=False))
        # prose chunks too (tagged) — without this, prose was lost on reload.
        for ch in self.prose:
            lines.append(json.dumps({"__prose__": True, **ch}, ensure_ascii=False))
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, text: str) -> "FactStore":
        import json
        store = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("__prose__"):
                    store.add_prose(obj.get("text", ""), obj.get("source", ""),
                                    obj.get("version", ""), obj.get("title", ""),
                                    source_id=obj.get("source_id", ""))
                else:
                    store.add(Fact(**obj))
            except Exception:
                continue
        return store
