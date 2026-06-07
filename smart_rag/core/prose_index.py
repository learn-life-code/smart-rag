#!/usr/bin/env python3
"""Semantic prose index — the 'how does it work' half of Smart RAG.

Facts answer "what is the value"; prose answers "how does it work". This embeds
the prose chunks (doc sections, code functions+comments, etc.) and retrieves them
BY MEANING — so "how does the deserializer work" finds the relevant section even
when the word never appears as a keyword. Reuses the same embedding engine as
attribute-intent (`distill/core/embed.py`), GPU/CPU/offline.

Degrades to keyword overlap if embeddings are unavailable — never worse than today.
"""
from __future__ import annotations

import re
from typing import List, Optional

from smart_rag.core import embed


class ProseIndex:
    """Embedded index over prose chunks [{text, source, version}]."""

    def __init__(self, chunks: List[dict], matrix=None):
        """chunks: [{text,title,source,...}]. If `matrix` (a precomputed, PERSISTED
        embedding matrix aligned to chunks) is given, use it — no re-embedding (the
        v2 'no rebuild on start' fix). Else embed now (in-memory mode)."""
        self.chunks = chunks
        self._matrix = matrix
        if self._matrix is None and chunks and embed.available():
            try:
                self._matrix = embed.embed([self._embed_text(c) for c in chunks])
            except Exception:
                self._matrix = None

    @staticmethod
    def _embed_text(chunk: dict) -> str:
        # Prepend the source/section so a chunk is self-describing to the embedder.
        head = chunk.get("title") or chunk.get("source", "")
        return (f"{head}\n{chunk.get('text','')}")[:2000]

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    # Cosine floor for ABSTENTION: below this, a "match" is noise — better to say
    # "I don't know" than return an irrelevant section. Calibrated for MiniLM where
    # a genuinely relevant section scores ~0.3+, unrelated ~0.1.
    MIN_RELEVANCE = 0.25

    def search(self, query: str, top_k: int = 5,
               min_relevance: "float | None" = None) -> List[dict]:
        """Return top-k relevant prose chunks ABOVE the relevance floor (semantic,
        else keyword). Returns [] when nothing is relevant enough — honest abstention."""
        if not self.chunks:
            return []
        floor = self.MIN_RELEVANCE if min_relevance is None else min_relevance
        if self.ready:
            import numpy as np
            qv = embed.embed_one(query)
            sims = self._matrix @ qv
            order = np.argsort(-sims)[:top_k]
            out = []
            for i in order:
                sc = float(sims[int(i)])
                if sc < floor:
                    continue   # abstain on weak matches
                c = dict(self.chunks[int(i)])
                c["score"] = sc
                out.append(c)
            return out
        return self._keyword(query, top_k)

    def _keyword(self, query: str, top_k: int) -> List[dict]:
        ql = {w for w in re.findall(r'\w+', query.lower()) if len(w) > 2}
        scored = []
        for c in self.chunks:
            t = c.get("text", "").lower()
            s = sum(1 for w in ql if w in t)
            if s:
                cc = dict(c); cc["score"] = s
                scored.append(cc)
        scored.sort(key=lambda c: -c["score"])
        return scored[:top_k]
