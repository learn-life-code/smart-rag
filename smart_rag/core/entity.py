#!/usr/bin/env python3
"""Entity recognition — what counts as the 'thing a fact is about'.

Generic by default (id-like tokens), with domain patterns registerable. The
entity is the row key in tabular data, the component/PID in logs, the party in a
contract. Keeping this pluggable is what lets Smart RAG work across domains.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional

# Default domain patterns (part numbers (configurable)) + a generic id-token fallback.
_ENTITY_PATTERNS: List[re.Pattern] = [
    re.compile(r'^\s*(75\d{8}|86\d{8})(-\w+)?\s*$'),
    re.compile(r'^\s*(F0\d{2}[.\s]?\d{3}[.\s]?\d{3})\s*$'),
]
# A looser "looks like an identifier" check for generic data.
_GENERIC_ID = re.compile(r'^[A-Za-z]{0,4}[-_]?\d{4,}[-_\w]*$')


def register_entity_pattern(pattern: str, flags=0) -> None:
    _ENTITY_PATTERNS.append(re.compile(pattern, flags))


def is_entity(value) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    if any(p.match(s) for p in _ENTITY_PATTERNS):
        return True
    return bool(_GENERIC_ID.match(s)) and len(s) <= 30


def is_strong_entity(value) -> bool:
    """Only domain-registered patterns (not the generic fallback)."""
    s = str(value or "").strip()
    return any(p.match(s) for p in _ENTITY_PATTERNS)


def entity_key(raw: str) -> str:
    """Normalize an entity to its canonical key (strip variant suffix)."""
    return re.split(r'[-\s]', str(raw).strip())[0]


def entity_column_of(rows: List[list], sample: int = 50) -> Optional[int]:
    """Find the column index whose cells most often look like entities (CSV/list)."""
    counts = defaultdict(int)
    for row in rows[:sample]:
        for i, cell in enumerate(row):
            if is_entity(cell):
                counts[i] += 1
    if counts:
        return max(counts, key=counts.get)
    # fallback: first column
    return 0 if rows and rows[0] else None
