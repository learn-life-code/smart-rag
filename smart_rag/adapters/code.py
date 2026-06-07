#!/usr/bin/env python3
"""Code/scripts adapter — source files → symbol FACTS + function/block PROSE.

Two outputs from one file:
  * facts: entity = symbol (function/class/def), attribute = kind|file|signature,
    value = the declaration → "which file defines deserializer_init" works.
  * prose: each function/block WITH its leading comment → "how does the deserializer
    work" finds the implementation + explanation by meaning.

Symbol extraction is heuristic regex (no compiler) — language-agnostic enough to be
useful across .py/.sh/.ps1/.c/.cpp/.h/.js/.ts/.java/.go/.rs. Honest about that.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List

from smart_rag.adapters.base import Adapter, _safe_relpath
from smart_rag.core.fact import Fact

# Per-language "definition" patterns → capture symbol name.
_DEF_PATTERNS = [
    re.compile(r'^\s*def\s+(\w+)\s*\(', re.M),                       # python
    re.compile(r'^\s*class\s+(\w+)', re.M),                          # python/others
    re.compile(r'^\s*function\s+([\w-]+)', re.M | re.I),             # ps1/js/sh
    re.compile(r'^\s*([\w-]+)\s*\(\)\s*\{', re.M),                   # sh function()
    re.compile(r'^\s*(?:[A-Za-z_][\w<>,\*\s]+?)\s+(\w+)\s*\([^;]*\)\s*\{', re.M),  # c/cpp/java func
    re.compile(r'^\s*(?:export\s+)?(?:async\s+)?(?:const|let|var)?\s*(\w+)\s*=\s*(?:async\s*)?\(', re.M),  # js arrow
    re.compile(r'^\s*func\s+(\w+)', re.M),                           # go
    re.compile(r'^\s*fn\s+(\w+)', re.M),                             # rust
]
_LANG = {".py": "python", ".sh": "shell", ".ps1": "powershell", ".c": "c",
         ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".java": "java",
         ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust",
         ".bat": "batch", ".pl": "perl", ".rb": "ruby",
         # Vector CAPL — C-like test logic (.can = CAPL program, .cin = include)
         ".can": "capl", ".cin": "capl"}


class CodeAdapter(Adapter):
    suffixes = tuple(_LANG.keys())
    name = "code"

    def extract(self, path: str) -> Iterable[Fact]:
        src = _safe_relpath(path)
        ext = os.path.splitext(path)[1].lower()
        lang = _LANG.get(ext, "code")
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            return
        for rx in _DEF_PATTERNS:
            for m in rx.finditer(text):
                name = m.group(1)
                if not name or len(name) < 2:
                    continue
                line = text[:m.start()].count("\n") + 1
                yield Fact(entity=name, attribute="defined_in", value=f"{src}:{line}",
                           source=src, span=m.group(0).strip()[:120])
                yield Fact(entity=name, attribute="kind", value=f"{lang} symbol", source=src)
        # CALL GRAPH (relation edges). Python via accurate stdlib AST; other
        # languages via the optional tree-sitter backend if installed (graceful).
        try:
            from smart_rag.adapters import callgraph as _cg
            yield from _cg.call_edges(text, src, ext)
        except Exception:  # noqa: BLE001
            pass

    def prose_chunks(self, path: str) -> Iterable[dict]:
        src = _safe_relpath(path)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            return
        # Chunk by function boundaries when detectable; else sliding window.
        bounds = sorted({m.start() for rx in _DEF_PATTERNS for m in rx.finditer(text)})
        if bounds:
            bounds.append(len(text))
            # include the comment lines directly above each function
            for i in range(len(bounds) - 1):
                start = self._with_leading_comment(text, bounds[i])
                seg = text[start:bounds[i + 1]].strip()
                if len(seg) > 30:
                    line = text[:bounds[i]].count("\n") + 1
                    yield {"text": seg[:1800], "source": src, "title": f"{src}:{line}"}
        else:
            for j in range(0, len(text), 1500):
                seg = text[j:j + 1500].strip()
                if len(seg) > 30:
                    yield {"text": seg, "source": src,
                           "title": f"{src} (block {j//1500 + 1})"}

    @staticmethod
    def _with_leading_comment(text: str, pos: int) -> int:
        """Walk backwards over comment/blank lines so a function's doc is included."""
        lines_before = text[:pos].split("\n")
        i = len(lines_before) - 1
        while i > 0:
            s = lines_before[i - 1].strip()
            if s.startswith(("#", "//", "*", "/*", '"""', "'''", "<#")) or s == "":
                i -= 1
            else:
                break
        return len("\n".join(lines_before[:i])) + (1 if i > 0 else 0)
