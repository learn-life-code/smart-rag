#!/usr/bin/env python3
"""Intent-aware resolution — find what the user MEANS, not just keyword strings.

Keyword search fails on intent: "how much storage" never matches the column
"UFS Size / GB". The fix exploits a fact store's structural advantage — the
ATTRIBUTE VOCABULARY is tiny (hundreds of names, not 130k chunks). So we embed
that small vocabulary once and resolve a query to the attribute(s) it MEANS, then
do the exact, grounded fact lookup.

Three layers:
  resolve_attribute(query)  → which column(s) the user means (storage→UFS,RAM)
  resolve_entity(query)     → which entity, described not named (HIGH NAR variant)
  resolve_value(query)      → reverse: which entities have a value (parts w/ SXM)

Everything degrades to keyword if embeddings are unavailable — never worse than
today, always offline-capable.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from smart_rag.core import embed

# Similarity thresholds (tunable). Close band = ambiguous → return several.
TAU_HIGH = 0.45     # confident single/primary match
TAU_LOW = 0.28      # below this, intent is too weak → fall back to keyword/full
CLOSE_BAND = 0.08   # within this of the top → treat as co-matches (storage→UFS+RAM)


class AttributeIndex:
    """Embedded index of a store's distinct attribute names. Tiny + fast."""

    def __init__(self, attributes: List[str]):
        self.attributes = list(dict.fromkeys(a for a in attributes if a))
        self._matrix = None
        if self.attributes and embed.available():
            try:
                self._matrix = embed.embed(self.attributes)
            except Exception:
                self._matrix = None

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def resolve(self, query: str, top_k: int = 6) -> List[Tuple[str, float]]:
        """Return [(attribute, score)] ranked. Empty if embeddings unavailable."""
        if not self.ready:
            return []
        import numpy as np
        qv = embed.embed_one(query)
        sims = self._matrix @ qv
        order = np.argsort(-sims)[:top_k]
        return [(self.attributes[i], float(sims[i])) for i in order]


def keyword_attribute_match(query: str, attributes: List[str]) -> List[str]:
    """Fallback: substring/word overlap between query and attribute names."""
    ql = set(re.findall(r'\w+', query.lower()))
    hits = []
    for a in attributes:
        al = set(re.findall(r'\w+', a.lower()))
        if ql & al:
            hits.append(a)
    return hits


def resolve_attribute(query: str, attr_index: Optional["AttributeIndex"],
                      attributes: List[str]) -> dict:
    """Resolve which attribute(s) a query means. Returns:
       {matched:[attr...], scores:{attr:score}, mode:"semantic|keyword|none", ambiguous:bool}
    """
    # Semantic first (the real fix).
    if attr_index and attr_index.ready:
        ranked = attr_index.resolve(query)
        if ranked and ranked[0][1] >= TAU_LOW:
            top = ranked[0][1]
            # co-matches within the close band AND above the low bar
            matched = [a for a, s in ranked if (top - s) <= CLOSE_BAND and s >= TAU_LOW]
            return {"matched": matched, "scores": dict(ranked),
                    "mode": "semantic", "ambiguous": len(matched) > 1,
                    "confident": ranked[0][1] >= TAU_HIGH}
    # Keyword fallback.
    kw = keyword_attribute_match(query, attributes)
    return {"matched": kw, "scores": {}, "mode": "keyword" if kw else "none",
            "ambiguous": len(kw) > 1, "confident": bool(kw)}


# Attributes that describe an entity (used to resolve descriptive entity queries).
_DESCRIPTOR_HINTS = ("variant", "region", "name", "type", "model", "plant", "platform")


def resolve_entity(query: str, store) -> List[str]:
    """Resolve a DESCRIPTIVE entity reference (no explicit id) → candidate entities.

    e.g. "the HIGH variant for NAR" → entities whose descriptor attrs match. Uses
    value matching on descriptor-like attributes. Returns candidate entity ids.
    """
    # Ignore generic words so "price of bitcoin" doesn't match every descriptor.
    _stop = {"the", "for", "and", "what", "which", "how", "does", "use", "with",
             "price", "value", "is", "are", "this", "that", "from", "have", "has",
             "work", "works", "logic", "about", "get", "show", "tell"}
    ql = [w for w in re.findall(r'\w+', query.lower())
          if len(w) > 2 and w not in _stop]
    if not ql:
        return []
    candidates = {}
    for ent in store.entities:
        facts = store.lookup(ent)
        score = 0
        for attr, rows in facts.items():
            if not any(h in attr.lower() for h in _DESCRIPTOR_HINTS):
                continue
            val = (rows[0]["value"] if rows else "").lower()
            # require WHOLE-WORD match in the descriptor value (not substring noise)
            vw = set(re.findall(r'\w+', val))
            score += sum(1 for w in ql if w in vw)
        if score:
            candidates[ent] = score
    # Require a minimum signal — a single weak match isn't an entity resolution.
    ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
    return [e for e, s in ranked if s >= max(1, (ranked[0][1] if ranked else 1) * 0.5)][:10]


def resolve_value(query: str, store, attr_index: Optional["AttributeIndex"]) -> List[dict]:
    """Reverse intent: 'which parts have SXM / use Winbond' → entities by value.

    Maps the query to likely attribute(s) (semantic), then finds entities whose
    value for those attributes matches the query's value terms.
    """
    # value term = the salient noun(s) after which/with/use/having
    m = re.search(r'(?:with|use[sd]?|having|has|contain[s]?)\s+(.+)$', query, re.I)
    value_term = (m.group(1).strip() if m else query).strip()
    # drop generic words; keep the SPECIFIC value tokens (brand/model/spec names)
    _stop = {"parts", "part", "which", "what", "list", "find", "the", "have",
             "has", "does", "with", "use", "uses", "used", "that", "for", "and"}
    value_words = [w for w in re.findall(r'\w+', value_term.lower())
                   if len(w) > 2 and w not in _stop]
    if not value_words:
        return []

    # Searching BY VALUE: scan ALL attributes (the user doesn't know the column,
    # and the value term — e.g. a brand "Winbond" — won't semantically match the
    # column name). Match the value token directly against every fact value.
    out = []
    for ent in store.entities:
        for attr, rows in store.lookup(ent).items():
            val = (rows[0]["value"] if rows else "").lower()
            if any(w in val for w in value_words):
                out.append({"entity": ent, "attribute": attr,
                            "value": rows[0]["value"]})
                break
    return out[:50]
