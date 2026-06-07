#!/usr/bin/env python3
"""Adaptive router — profile the query, choose facts vs prose (the 'adaptable' core).

Smart RAG keeps BOTH a fact store (entity-attribute lookups) and prose chunks
(narrative). The router reads the query and decides which surface answers it:

  * entity + attribute ("UFS for SKU1001")   → FACT lookup (grounded, instant)
  * conceptual ("how is release handled")        → PROSE / semantic
  * mixed                                        → facts + prose context

This is per-QUERY adaptation; corpus-shape profiling (which strategy a whole
dataset needs) lives in profile.py.
"""
from __future__ import annotations

import re
from typing import Optional

from smart_rag.core.entity import is_entity

_ENTITY_RE = re.compile(r'\b(75\d{8}|86\d{8}|F0\d{2}[.\s]?\d{3}[.\s]?\d{3}|[A-Za-z]{2,}/\w+)\b')
_ATTR_WORDS = ("ufs", "ram", "memory", "sxm", "pcb", "part number", "pn", "version",
               "plant", "variant", "tuner", "gnss", "ethernet", "supplier", "ecr",
               "ppap", "size", "config", "error", "errors", "crash", "timeout",
               "warning", "status", "date", "supplier")
_CONCEPT_WORDS = ("how", "why", "process", "explain", "strategy", "history", "overview",
                  "approach", "compare", "difference", "summary", "describe")


def classify_query(query: str) -> str:
    q = query.lower()
    has_entity = bool(_ENTITY_RE.search(query))
    has_attr = any(w in q for w in _ATTR_WORDS)
    has_concept = any(w in q for w in _CONCEPT_WORDS)
    if has_entity and has_attr and not has_concept:
        return "lookup"
    if has_concept and not (has_entity and has_attr):
        return "concept"
    return "mixed"


def extract_entity(query: str) -> Optional[str]:
    m = _ENTITY_RE.search(query)
    return m.group(0) if m else None


def attr_hint(query: str) -> Optional[str]:
    q = query.lower()
    for w in _ATTR_WORDS:
        if w in q:
            return w
    return None
