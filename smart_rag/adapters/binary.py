#!/usr/bin/env python3
"""Binary adapter — built-in ELF symbol extraction (so Smart RAG handles compiled
objects itself, no separate codegraph tool to hunt for).

Covers the COMMON case directly:
  * .so / .elf / .o  → symbols via pyelftools (the open ELF standard): function +
    object symbols become entity facts (entity=symbol, kind=elf function/object,
    defined_in=file). Optional `prose_chunks` pulls human-readable strings.
  * Falls back to a stdlib `strings`-style scan if pyelftools is absent.

What it does NOT do (still needs a specialized binary extractor / codegraph):
  * Qualcomm MBN, sparse Android .img, packed firmware — proprietary container
    formats. For those, the index manager advises running codegraph first.

So: ELF objects are now first-class in Smart RAG; exotic firmware is where you still
reach for codegraph (and Smart RAG tells you when).
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

_ELF_MAGIC = b"\x7fELF"
_PRINTABLE = re.compile(rb"[\x20-\x7e]{5,}")   # ascii runs ≥5 chars


class BinaryAdapter(Adapter):
    suffixes = (".so", ".elf", ".o")
    name = "binary"
    emits = ("elf_function", "elf_object")
    standard = "ELF (System V ABI)"

    def can_handle(self, path: str) -> bool:
        if path.lower().endswith(self.suffixes):
            return True
        # also accept extensionless ELF files by magic sniff
        try:
            with open(path, "rb") as fh:
                return fh.read(4) == _ELF_MAGIC
        except Exception:
            return False

    def extract(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        try:
            from elftools.elf.elffile import ELFFile
        except Exception:
            return   # pyelftools absent → no symbol facts (strings still via prose)
        try:
            n_syms = 0
            with open(path, "rb") as fh:
                elf = ELFFile(fh)
                for sect in elf.iter_sections():
                    if not sect.name.startswith((".symtab", ".dynsym")):
                        continue
                    for sym in sect.iter_symbols():
                        nm = sym.name
                        if not nm or len(nm) < 2:
                            continue
                        info = sym["st_info"]["type"]
                        if info == "STT_FUNC":
                            n_syms += 1
                            yield Fact(entity=nm, attribute="autosar_type",
                                       value="elf_function", source=src,
                                       kind="extracted")
                            yield Fact(entity=nm, attribute="defined_in",
                                       value=src, source=src)
                        elif info == "STT_OBJECT":
                            n_syms += 1
                            yield Fact(entity=nm, attribute="autosar_type",
                                       value="elf_object", source=src,
                                       kind="extracted")
            # STRIPPED ELF (no symbols, e.g. production firmware) → record a marker
            # fact so the index manager can advise running codegraph for it.
            if n_syms == 0:
                yield Fact(entity=src, attribute="binary_stripped",
                           value="no symbol table — needs a binary extractor (codegraph)",
                           source=src, kind="extracted")
        except Exception:
            return

    def prose_chunks(self, path: str) -> Iterable[dict]:
        """Human-readable strings (error messages, paths, versions) — searchable
        even without symbols. Capped so a huge binary doesn't flood the index."""
        src = os.path.basename(path)
        try:
            data = open(path, "rb").read(8_000_000)   # cap at 8 MB scan
        except Exception:
            return
        strs = [m.group().decode("ascii", "replace")
                for m in _PRINTABLE.finditer(data)]
        # keep informative ones (letters present), dedup, cap
        seen = set()
        keep = []
        for s in strs:
            if any(c.isalpha() for c in s) and s not in seen:
                seen.add(s)
                keep.append(s)
            if len(keep) >= 4000:
                break
        for i in range(0, len(keep), 60):
            chunk = "\n".join(keep[i:i + 60])
            if chunk.strip():
                yield {"text": chunk, "source": src,
                       "title": f"{src} strings ({i}-{i+60})"}
