#!/usr/bin/env python3
"""Logs adapter — DLT / logcat / generic text logs → component-fact-time view.

The software-engineer Stage-1 case. A log is thousands of lines, mostly noise and
repetition. Smart RAG turns it into an ENTITY-FACT view: entity = the component /
PID / tag emitting the line; facts = the notable events (errors, warnings, state
changes, NRCs, hex codes) with their timestamp. So instead of grepping 200k lines,
you ask "what errors did component X have, and when" and get a deduplicated,
time-ordered, source-cited answer.

Recognized line shapes (extend freely):
  logcat:  MM-DD HH:MM:SS.mmm  PID TID  L  TAG: message
  DLT:     H:MM:SS.s  [LEVEL] [APPID] CTX/SUB  message
  generic: any line with a [COMPONENT]/word: message and a severity keyword
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

_LOGCAT = re.compile(
    r'^(?P<ts>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\d+\s+\d+\s+'
    r'(?P<lvl>[VDIWEF])\s+(?P<tag>[^:]{1,60}):\s*(?P<msg>.*)$')
_DLT = re.compile(
    r'^(?P<ts>\d+:\d{2}:\d{2}(?:\.\d+)?)\s+(?:\[(?P<lvl>[A-Z]+)\]\s+)?'
    r'(?:\[(?P<app>[^\]]+)\]\s+)?(?P<ctx>\S+/\S+)?\s*(?P<msg>.*)$')
_GENERIC = re.compile(r'(?P<comp>[A-Za-z][\w.]{2,40})(?:::\w+)?\s*[:\-]\s*(?P<msg>.{4,})$')

_SEVERITY = re.compile(
    # No trailing \b → stems match inflections ("fail"→"failed", "error"→"errors",
    # "reject"→"rejected"). The earlier \b dropped legitimate error lines.
    r'\b(error|fail|fatal|exception|crash|abort|timeout|denied|reject|panic|'
    r'segfault|sigabrt|sigsegv|warning|warn|nrc|0x[0-9a-f]{2,})', re.I)

_EVENT_KIND = [
    (re.compile(r'\b(fatal|panic|crash|abort|sigabrt|sigsegv|segfault|tombstone)', re.I), "crash"),
    (re.compile(r'\b(error|fail|exception|denied|reject)', re.I), "error"),
    (re.compile(r'\b(timeout|stall|hang|blocked)', re.I), "timeout"),
    (re.compile(r'\b(warning|warn)\b', re.I), "warning"),
    (re.compile(r'\bNRC\b|\bnegative response\b', re.I), "uds_nrc"),
    (re.compile(r'\b0x[0-9a-f]{2,}\b', re.I), "hex_code"),
]


def _classify(msg: str) -> str:
    for rx, kind in _EVENT_KIND:
        if rx.search(msg):
            return kind
    return "event"


class LogsAdapter(Adapter):
    suffixes = (".log", ".txt", ".dlt", ".logcat", ".slog", ".trace", ".out")
    name = "logs"

    def extract(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        ver = self._run_id(src)
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except Exception:
            return
        seen = set()   # (component, kind, normalized-msg) → dedup repetitive logs
        with fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                comp, ts, msg = self._parse(line)
                if not comp or not msg:
                    continue
                # Only keep NOTABLE lines (severity/events) — that's the distillation.
                if not _SEVERITY.search(msg):
                    continue
                kind = _classify(msg)
                norm = re.sub(r'\b(0x[0-9a-f]+|\d{3,})\b', '#', msg.lower())[:120]
                key = (comp, kind, norm)
                if key in seen:
                    continue
                seen.add(key)
                yield Fact(entity=comp, attribute=kind, value=msg[:300],
                           source=f"{src}:L{i+1}", version=ver, date=self._iso(ts),
                           span=line[:200])

    def prose_chunks(self, path: str) -> Iterable[dict]:
        return []   # logs are fact-only

    @staticmethod
    def _parse(line: str):
        m = _LOGCAT.match(line)
        if m:
            return m.group("tag").strip(), m.group("ts"), m.group("msg").strip()
        m = _DLT.match(line)
        if m and (m.group("ctx") or m.group("app")):
            comp = (m.group("ctx") or m.group("app") or "").strip()
            return comp, m.group("ts"), (m.group("msg") or "").strip()
        m = _GENERIC.search(line)
        if m:
            return m.group("comp").strip(), "", m.group("msg").strip()
        return None, None, None

    @staticmethod
    def _iso(ts) -> str:
        # logcat "MM-DD HH:MM:SS.mmm" has no year; keep time-of-day for ordering.
        return (ts or "")[:19]

    @staticmethod
    def _run_id(name: str) -> str:
        m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', name)
        return m.group(1) if m else name
