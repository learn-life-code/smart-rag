#!/usr/bin/env python3
"""Hybrid retrieval + single-scale rerank (v2 — the cure for the hijack class).

Given a QueryPlan, gather candidates from ALL channels and score them on ONE
normalized scale, then rerank. Because every candidate competes on the same scale,
a weak entity match can NEVER outrank a strong prose hit — which is precisely the
bug the old ad-hoc routing kept producing.

Channels:
  * EXACT fact   — entity + (intent-resolved) attribute  → high, precise score
  * VECTOR prose — semantic similarity over persisted embeddings (cosine)
  * KEYWORD/FTS  — BM25-ish term overlap over facts + prose
Each channel emits Candidates with a raw score; we min-max normalize PER CHANNEL,
weight by channel reliability, and sort. Calibrated thresholds drive abstention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Candidate:
    text: str
    source: str
    score: float            # final normalized [0,1] score (for ranking)
    channel: str            # exact | vector | keyword
    kind: str = "prose"     # fact | prose
    entity: str = ""
    attribute: str = ""
    version: str = ""
    abs_relevance: float = 0.0   # ABSOLUTE relevance (cosine / exact=1) for ABSTENTION


# Channel weights — exact lookups are most reliable, then vector, then keyword.
_W = {"exact": 1.0, "vector": 0.85, "keyword": 0.55}

# CALIBRATED thresholds (measured on a labeled F:\Test set, not guessed — the
# review's point). Observed: real-topic answers land abs_relevance 0.35-0.53;
# unrelated junk lands 0.18-0.20. Clean gap at ~0.20-0.35 → set the floor in it.
#   < REL_FLOOR        → NOT_FOUND (abstain)
#   REL_FLOOR..MED_BAR → LOW / INSUFFICIENT_EVIDENCE
#   MED_BAR..HIGH_BAR  → MEDIUM
#   >= HIGH_BAR        → HIGH
# Re-tune via scripts/eval.py against the labeled set, never hand-pick.
REL_FLOOR = 0.27
MED_BAR = 0.33
HIGH_BAR = 0.50


def _norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def hybrid_retrieve(plan, store, *, top_k: int = 8) -> List[Candidate]:
    """Return reranked Candidates for a QueryPlan. No LLM."""
    cands: List[Candidate] = []

    # ── EXACT: entity + intent-resolved attribute(s) ─────────────────────────
    if plan.entity and plan.intent in ("FACT", "MIXED"):
        cands += _exact_facts(plan, store)

    # ── VECTOR: semantic prose (PROSE/MIXED, and as backstop for FACT) ───────
    if plan.intent in ("PROSE", "MIXED", "FACT"):
        cands += _vector_prose(plan, store, top_k)

    # ── KEYWORD: FTS/term overlap over facts + prose (always, as a channel) ──
    cands += _keyword(plan, store, top_k)

    if not cands:
        return []

    # RANK on a normalized+weighted scale, but PRESERVE abs_relevance (set per
    # channel) for abstention — normalization alone makes even a weak top hit look
    # strong, so abstention must use the ABSOLUTE signal (cosine / exact match).
    by_ch: dict = {}
    for c in cands:
        by_ch.setdefault(c.channel, []).append(c)
    final: List[Candidate] = []
    for ch, group in by_ch.items():
        ns = _norm([c.score for c in group])
        for c, n in zip(group, ns):
            c.score = n * _W.get(ch, 0.5)   # for ranking order
            final.append(c)

    seen = {}
    for c in final:
        key = (c.source, c.text[:60])
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    ranked = sorted(seen.values(), key=lambda c: -c.score)
    return ranked[:top_k]


def _exact_facts(plan, store) -> List[Candidate]:
    from smart_rag.core import intent as _intent
    out: List[Candidate] = []
    store.ensure_attr_index()
    res = _intent.resolve_attribute(plan.raw, store.attr_index,
                                    store.distinct_attributes())
    matched = res.get("matched") or []
    if matched:
        for attr in matched:
            rows = store.lookup(plan.entity, attr).get(attr)
            if rows:
                r0 = rows[0]
                out.append(Candidate(
                    text=f"{attr}: {r0['value']}", source=(r0['sources'] or ['?'])[0],
                    score=1.0, channel="exact", kind="fact",
                    entity=plan.entity, attribute=attr,
                    version=(r0['versions'] or [''])[0], abs_relevance=1.0))
    else:
        for attr, rows in list(store.lookup(plan.entity).items())[:8]:
            out.append(Candidate(
                text=f"{attr}: {rows[0]['value']}", source=(rows[0]['sources'] or ['?'])[0],
                score=0.6, channel="exact", kind="fact",
                entity=plan.entity, attribute=attr, abs_relevance=0.7))
    return out


def _vector_prose(plan, store, top_k: int) -> List[Candidate]:
    idx = store.ensure_prose_index()
    if not idx or not getattr(idx, "ready", False):
        return []
    out: List[Candidate] = []
    for h in idx.search(plan.raw, top_k=top_k, min_relevance=0.0):  # floor applied later
        cos = float(h.get("score", 0.0))
        out.append(Candidate(
            text=h["text"][:600], source=h.get("title") or h.get("source", ""),
            score=cos, channel="vector", kind="prose", abs_relevance=cos))
    return out


def _keyword(plan, store, top_k: int) -> List[Candidate]:
    """Term-overlap over facts + prose (no FTS dependency for the hot layer).
    Requires SPECIFIC topic terms (stopwords already removed in the plan)."""
    terms = [t for t in plan.topic_terms if len(t) > 2]
    if not terms:
        return []
    out: List[Candidate] = []
    # prose term overlap
    for ch in store.prose[:5000]:
        t = ch.get("text", "").lower()
        score = sum(1 for w in terms if w in t)
        if score:
            # keyword overlap is WEAK evidence: abs_relevance scales with the
            # FRACTION of query terms matched, capped low so it can't alone clear
            # the abstention bar (prevents "price"→"Cost" false ANSWERED).
            frac = score / max(len(terms), 1)
            out.append(Candidate(text=ch["text"][:500],
                                  source=ch.get("title") or ch.get("source", ""),
                                  score=float(score), channel="keyword", kind="prose",
                                  abs_relevance=min(0.35, 0.18 * score) * frac))
    out.sort(key=lambda c: -c.score)
    out = out[:top_k]
    # fact term overlap (entities/attrs/values)
    fout: List[Candidate] = []
    for ent in store.entities:
        el = ent.lower()
        hit_e = any(w in el for w in terms)
        for attr, rows in store.lookup(ent).items():
            hay = (attr + " " + rows[0]["value"]).lower()
            s = (2 if hit_e else 0) + sum(1 for w in terms if w in hay)
            if s:
                frac = s / max(len(terms) + (2 if hit_e else 0), 1)
                fout.append(Candidate(
                    text=f"{ent} · {attr}: {rows[0]['value']}",
                    source=(rows[0]["sources"] or ["?"])[0], score=float(s),
                    channel="keyword", kind="fact", entity=ent, attribute=attr,
                    abs_relevance=min(0.35, 0.18 * s) * frac))
        if len(fout) > top_k * 4:
            break
    fout.sort(key=lambda c: -c.score)
    return out + fout[:top_k]
