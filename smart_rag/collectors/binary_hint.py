#!/usr/bin/env python3
"""Detect binary/firmware files Smart RAG can't fully parse, and advise running
codegraph first.

Smart RAG reads TEXT + structured formats. It cannot extract symbols/call-edges
from a compiled binary (.so/.elf/.exe) or a firmware image (.img/.mbn/.bin) — that
needs a binary symbol extractor (codegraph). When a folder has these, we tell the
user: run codegraph first, then Smart RAG absorbs the resulting codegraph.db.
"""
from __future__ import annotations

import os
from typing import List

# Binary / firmware extensions that require a binary symbol extractor (codegraph).
_BINARY_EXT = {
    ".a", ".lib", ".dll", ".dylib",   # static/win libs (ELF .so/.elf/.o handled by binary adapter)
    ".bin", ".img", ".mbn", ".hex", ".fls", ".srec",        # firmware images
    ".dex", ".odex", ".vdex", ".oat", ".apk",               # android
    ".ko", ".sys",                                          # kernel modules
}
# A folder is "a build" if it has a codegraph DB or these firmware markers.
_BUILD_MARKERS = {"NON-HLOS.bin", "abl.elf", "boot.img", "system.img"}


def scan_for_binaries(paths: List[str], *, sample: int = 50) -> dict:
    """Return {has_binaries, count, kinds, has_codegraph, advice} for a file list."""
    kinds: dict = {}
    has_codegraph = False
    for p in paths:
        low = p.lower()
        if low.endswith(".db") and os.path.basename(low) in (
                "codegraph.db",) or "/.codegraph/" in low.replace("\\", "/"):
            has_codegraph = True
        ext = os.path.splitext(low)[1]
        if ext in _BINARY_EXT:
            kinds[ext] = kinds.get(ext, 0) + 1
    count = sum(kinds.values())
    advice = ""
    if count and not has_codegraph:
        top = ", ".join(f"{n}{e}" for e, n in
                        sorted(kinds.items(), key=lambda x: -x[1])[:4])
        advice = (
            f"⚠ Found {count} binary/firmware file(s) ({top}) that Smart RAG can't "
            f"extract symbols from. For code structure (symbols, call graph) in these, "
            f"RUN CODEGRAPH FIRST:  codegraph index <folder>  (or "
            f"binary_symbol_extractor), then re-run Smart RAG — it will absorb the "
            f"codegraph.db automatically.")
    return {"has_binaries": bool(count), "count": count, "kinds": kinds,
            "has_codegraph": has_codegraph, "advice": advice}
