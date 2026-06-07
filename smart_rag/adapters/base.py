#!/usr/bin/env python3
"""Adapter contract — the extensibility that makes Smart RAG work on ANY format.

A format adapter turns one source (a file) into Facts (+ optional prose chunks).
Adding a new format = adding one adapter; the core never changes. This is what
lets the office point it at their own tools' data.

    class MyAdapter(Adapter):
        suffixes = (".xyz",)
        def extract(self, path) -> Iterable[Fact]: ...
        def prose_chunks(self, path) -> Iterable[dict]: ...   # optional

Register adapters in adapters/__init__.py; the API dispatches by file suffix.
"""
from __future__ import annotations

import os
from typing import Iterable, List

from smart_rag.core.fact import Fact


def _safe_relpath(path: str) -> str:
    """relpath that never crashes across drives (F: vs C: on Windows)."""
    try:
        rp = os.path.relpath(path)
        # if relpath had to climb out a lot, prefer a cleaner tail
        return rp if not rp.startswith("..") else path
    except ValueError:
        return path


class Adapter:
    """Base format adapter. Subclasses set `suffixes` and implement `extract`."""

    suffixes: tuple = ()
    name: str = "base"

    def can_handle(self, path: str) -> bool:
        p = path.lower()
        return any(p.endswith(s) for s in self.suffixes)

    def extract(self, path: str) -> Iterable[Fact]:
        """Yield Facts from the source. Must be overridden."""
        raise NotImplementedError

    def prose_chunks(self, path: str) -> Iterable[dict]:
        """Yield {text, source, version} for narrative content. Optional."""
        return []
