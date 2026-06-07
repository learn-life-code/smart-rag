#!/usr/bin/env python3
"""Docs adapter — Markdown / text / DOCX / PDF → semantic prose chunks.

Captures EXPLANATORY content (how things work) that fact-lookup misses. Chunks are
header-aware (section titles kept) so each chunk is self-describing and retrieves
well by meaning. Emits prose only (no facts) — the prose index searches these.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List

from smart_rag.adapters.base import Adapter, _safe_relpath
from smart_rag.core.fact import Fact

_HEADER = re.compile(r'^(#{1,6})\s+(.*)$')


class DocsAdapter(Adapter):
    suffixes = (".md", ".rst", ".docx", ".pdf", ".doc", ".markdown",
                ".html", ".htm", ".xvp")   # html docs + Vector panel XML (text)
    name = "docs"

    def extract(self, path: str) -> Iterable[Fact]:
        return []   # docs are prose, not facts

    def prose_chunks(self, path: str) -> Iterable[dict]:
        low = path.lower()
        src = _safe_relpath(path)
        try:
            if low.endswith((".md", ".rst", ".markdown")):
                text = open(path, encoding="utf-8", errors="replace").read()
                yield from self._chunk_markdown(text, src)
            elif low.endswith(".docx"):
                yield from self._chunk_docx(path, src)
            elif low.endswith(".pdf"):
                yield from self._chunk_pdf(path, src)
            elif low.endswith((".html", ".htm", ".xvp")):
                # strip tags → plain text, then chunk like markdown
                raw = open(path, encoding="utf-8", errors="replace").read()
                text = re.sub(r'<[^>]+>', ' ', raw)
                text = re.sub(r'&\w+;', ' ', text)
                text = re.sub(r'[ \t]+', ' ', text)
                yield from self._chunk_markdown(text, src)
            else:
                # PLAIN TEXT (.txt routed here as a doc, .rst, or any other text):
                # chunk it like prose. Previously this fell through → 0 chunks, so a
                # valid .txt document was wrongly reported 'empty'.
                text = open(path, encoding="utf-8", errors="replace").read()
                yield from self._chunk_markdown(text, src)
        except Exception as e:  # noqa: BLE001
            print(f"[docs] {os.path.basename(path)}: {e}")

    # ── markdown / text: split on headers, keep the section title ────────────
    def _chunk_markdown(self, text: str, src: str) -> Iterable[dict]:
        # strip YAML front-matter but keep it as a chunk (skill descriptions live there)
        fm = ""
        m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
        if m:
            fm = m.group(1)
            text = text[m.end():]
            yield {"text": fm, "source": src, "title": f"{src} (front-matter)"}
        cur_title = src
        buf: List[str] = []
        for line in text.splitlines():
            h = _HEADER.match(line)
            if h:
                if buf and any(b.strip() for b in buf):
                    yield {"text": "\n".join(buf).strip(), "source": src, "title": cur_title}
                cur_title = f"{src} › {h.group(2).strip()}"
                buf = [line]
            else:
                buf.append(line)
            if sum(len(b) for b in buf) > 1500:   # cap section size
                yield {"text": "\n".join(buf).strip(), "source": src, "title": cur_title}
                buf = []
        if buf and any(b.strip() for b in buf):
            yield {"text": "\n".join(buf).strip(), "source": src, "title": cur_title}

    def _chunk_docx(self, path: str, src: str) -> Iterable[dict]:
        from docx import Document
        doc = Document(path)
        buf: List[str] = []
        title = src
        for p in doc.paragraphs:
            t = p.text.strip()
            if not t:
                continue
            if (p.style and p.style.name and "head" in p.style.name.lower()):
                if buf:
                    yield {"text": "\n".join(buf), "source": src, "title": title}
                    buf = []
                title = f"{src} › {t}"
            buf.append(t)
            if sum(len(b) for b in buf) > 1500:
                yield {"text": "\n".join(buf), "source": src, "title": title}; buf = []
        if buf:
            yield {"text": "\n".join(buf), "source": src, "title": title}

    def _chunk_pdf(self, path: str, src: str) -> Iterable[dict]:
        import pymupdf
        doc = pymupdf.open(path)
        for i, page in enumerate(doc):
            t = page.get_text().strip()
            if t:
                # split long pages into ~1500-char chunks
                for j in range(0, len(t), 1500):
                    yield {"text": t[j:j + 1500], "source": src,
                           "title": f"{src} › p{i+1}"}
        doc.close()
