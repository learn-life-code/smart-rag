#!/usr/bin/env python3
"""Corpus-shape profiler — measure the data, recommend a strategy (the 'adaptive' meta).

Different data wants different handling. Profile measures: redundancy, tabular vs
prose mix, versioned families, entity density — and recommends DISTILL / TOON /
SEMANTIC / HYBRID. This is what makes Smart RAG adapt to ANY corpus instead of
forcing one pipeline.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

_ENTITY_RE = re.compile(r'\b(75\d{8}|86\d{8}|F0\d{2}[.\s]?\d{3}[.\s]?\d{3})\b')
_VER_RE = re.compile(r'[_\s\-]v?\d+(\.\d+)?\.(xls[xm]?|pdf|docx|csv|json|log)$', re.I)


@dataclass
class Profile:
    n_items: int
    redundancy: float
    tabular_fraction: float
    versioned_families: int
    entity_density: float
    recommended: str = ""

    def summary(self) -> str:
        return (f"items={self.n_items:,} redundancy={self.redundancy:.0%} "
                f"tabular={self.tabular_fraction:.0%} versioned={self.versioned_families} "
                f"entity_density={self.entity_density:.3f} → {self.recommended.upper()}")


def profile_items(items: List[Dict[str, Any]]) -> Profile:
    """items: [{text, source}] (chunks) OR can be derived from a FactStore."""
    n = len(items) or 1
    pref = Counter((it.get("text", "") or "")[:200] for it in items)
    redundancy = sum(v for v in pref.values() if v > 1) / n
    tab = sum(1 for it in items
              if "Columns:" in (it.get("text", "") or "")
              or (it.get("metadata") and any(k != "sheet" for k in it["metadata"])))
    tabular_fraction = tab / n
    fams = defaultdict(int)
    for it in items:
        base = _VER_RE.sub("", it.get("source", ""))
        fams[re.sub(r'\.(xls[xm]?|pdf|docx|csv|json|log)$', "", base)] += 1
    versioned = sum(1 for v in fams.values() if v >= 3)
    ents = set()
    for it in items:
        ents.update(_ENTITY_RE.findall(it.get("text", "") or ""))
    entity_density = len(ents) / n

    if redundancy > 0.5 and tabular_fraction > 0.4 and entity_density > 0.003:
        rec = "distill"
    elif tabular_fraction > 0.6:
        rec = "toon"
    elif tabular_fraction < 0.3:
        rec = "semantic"
    else:
        rec = "hybrid"
    return Profile(n, redundancy, tabular_fraction, versioned, entity_density, rec)
