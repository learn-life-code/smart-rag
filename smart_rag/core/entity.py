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

# Domain patterns are OPTIONAL hints (register your own via register_entity_pattern).
# By default Smart RAG is domain-agnostic: a STRONG entity is any id-like token —
# letters+digits (SKU1042, PART-0099, ABC123), a long numeric (7515401998), or a
# dotted/structured code. This is what makes it work out-of-the-box on any data.
_ENTITY_PATTERNS: List[re.Pattern] = []   # user-registered domain patterns (optional)

# Generic "looks like an identifier": an alphanumeric id with at least one digit run
# (so it's not a plain word), optionally with -/_/. separators.
_GENERIC_ID = re.compile(r'^[A-Za-z]{0,6}[-_]?\d{3,}[-_.\w]*$')
# Also accept letter-led codes with an embedded number (e.g. ECU_A1, NMOS3, P0420).
_CODE_ID = re.compile(r'^[A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*$')


def register_entity_pattern(pattern: str, flags=0) -> None:
    _ENTITY_PATTERNS.append(re.compile(pattern, flags))


def is_entity(value) -> bool:
    s = str(value or "").strip()
    if not s or len(s) > 40:
        return False
    if any(p.match(s) for p in _ENTITY_PATTERNS):
        return True
    return bool(_GENERIC_ID.match(s) or _CODE_ID.match(s))


def is_strong_entity(value) -> bool:
    """An id-like token: a registered domain pattern OR a generic alphanumeric id.
    Domain-agnostic by default so the tabular/chip/etc adapters find the entity
    column on ANY data (SKU1042, P0420, 7515401998), not just one company's PNs."""
    s = str(value or "").strip()
    if any(p.match(s) for p in _ENTITY_PATTERNS):
        return True
    # require a digit run (excludes plain words) and reasonable length
    return (bool(_GENERIC_ID.match(s)) or bool(_CODE_ID.match(s))) and 2 <= len(s) <= 40


# A variant suffix to strip: a SHORT trailing token after the LAST separator that
# looks like a revision marker (B1, p0, rev2, -A). We keep the full id otherwise, so
# generic hyphenated ids (PART-0099) are NOT truncated to 'PART'.
_VARIANT_SUFFIX = re.compile(r'^[-\s]([A-Za-z]?\d{0,2}[A-Za-z]?\d{0,2})$')


def entity_key(raw: str) -> str:
    """Canonical entity key. Strips a trailing VARIANT suffix only when the id has a
    long stable stem (so 7515401998-B1 → 7515401998) but preserves general ids like
    PART-0099 or SER-100 intact."""
    s = str(raw).strip()
    # split on the LAST separator; only strip if the stem is clearly the identity
    m = re.match(r'^(.*?)([-\s][A-Za-z0-9]{1,3})$', s)
    if m:
        stem, suffix = m.group(1), m.group(2)
        # strip only if the stem itself already looks like a complete id (has 4+ digits)
        if len(re.sub(r'\D', '', stem)) >= 4 and _VARIANT_SUFFIX.match(suffix):
            return stem
    return s


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
