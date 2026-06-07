#!/usr/bin/env python3
"""Domain value decoders — turn coded values into readable facts.

Real engineering data hides values inside compound codes (e.g. a config string
"...SA8255P36n810H-UFS256S-SAILFlM..."). A plain RAG returns the whole opaque
string; Smart RAG decodes the meaningful part. Decoders are small, additive rules —
extend per domain. Each returns a friendlier string, or the input unchanged.
"""
from __future__ import annotations

import re
from typing import Callable, List, Tuple

_DECODERS: List[Tuple[re.Pattern, Callable]] = [
    (re.compile(r'\bUFS\s?(\d+)\b', re.I), lambda m: f"{m.group(1)} GB UFS"),
    (re.compile(r'\bDDR(\d)\b', re.I), lambda m: f"DDR{m.group(1)}"),
]


def decode_value(value: str) -> str:
    """Return a decoded value if a rule matches a CODE form, else the value as-is.

    Only rewrites when the value looks like a bare code (short, no spaces) to avoid
    mangling normal prose values.
    """
    v = (value or "").strip()
    if not v or " " in v and len(v) > 40:
        return v
    for rx, fn in _DECODERS:
        m = rx.search(v)
        # only rewrite when the match IS essentially the whole short token
        if m and (len(v) <= 12 or v.upper().startswith(m.group(0).upper())):
            return fn(m)
    return v


def register_decoder(pattern: str, fn: Callable, flags=re.I) -> None:
    """Let a domain add its own decoder at runtime."""
    _DECODERS.append((re.compile(pattern, flags), fn))
