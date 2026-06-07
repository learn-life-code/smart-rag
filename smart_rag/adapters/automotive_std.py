#!/usr/bin/env python3
"""Automotive standards adapter — ODX (ISO 22901) + A2L (ASAM MCD-2 MC).

These join the existing DBC/ARXML coverage so Smart RAG speaks the automotive
diagnostics + calibration languages:

  ODX (.odx, .pdx, .odx-d/-c/-f)  — ECU diagnostics: DTCs, services, DIDs.
      DTC        → entity, attributes: code, display_code, text
      DIAG-SERVICE / DID → entity, attributes + in_layer relation
  A2L (.a2l)  — ECU measurement/calibration variables (XCP/CCP).
      MEASUREMENT / CHARACTERISTIC → entity, attributes: type, unit, address,
      conversion; relation to its COMPU_METHOD.

Schema-aware: only the standard's known elements are extracted (no arbitrary XML
flattening). These are ASAM/ISO interchange formats — structured by design, which
is exactly what the fact/relation model wants.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact


class AutomotiveStdAdapter(Adapter):
    suffixes = (".odx", ".pdx", ".odx-d", ".odx-c", ".odx-f", ".a2l")
    name = "automotive_std"
    emits = ("dtc", "diag_service", "did", "measurement", "characteristic", "compu_method")
    standard = "ISO 22901 (ODX) + ASAM MCD-2 MC (A2L)"

    def extract(self, path: str) -> Iterable[Fact]:
        low = path.lower()
        if low.endswith(".a2l"):
            yield from self._a2l(path)
        else:
            yield from self._odx(path)

    # ── ODX (ISO 22901) — diagnostics ────────────────────────────────────────
    def _odx(self, path: str) -> Iterable[Fact]:
        try:
            import xml.etree.ElementTree as ET
        except Exception:
            return
        src = os.path.basename(path)
        try:
            it = ET.iterparse(path, events=("start", "end"))
        except Exception:
            return
        # element types ODX defines that we extract (schema-aware allowlist)
        _DTC = "DTC"
        _SVC = "DIAG-SERVICE"
        _DID = "DATA-OBJECT-PROP"
        stack = []   # [(type, short_name|None)]
        cur_text_attr = None
        try:
            for ev, el in it:
                tag = el.tag.split("}")[-1]
                if ev == "start":
                    if tag in (_DTC, _SVC, _DID):
                        stack.append([tag, None])
                    continue
                owner = stack[-1] if stack else None
                if tag == "SHORT-NAME" and el.text and owner and owner[1] is None:
                    owner[1] = el.text.strip()
                    kind = {"DTC": "dtc", "DIAG-SERVICE": "diag_service",
                            "DATA-OBJECT-PROP": "did"}[owner[0]]
                    yield Fact(entity=owner[1], attribute="autosar_type",
                               value=kind, source=src, kind="extracted")
                elif owner and owner[1]:
                    if tag in ("TROUBLE-CODE", "DISPLAY-TROUBLE-CODE") and el.text:
                        yield Fact(entity=owner[1], attribute=tag.replace("-", "_").lower(),
                                   value=el.text.strip(), source=src)
                    elif tag == "TEXT" and el.text and el.text.strip():
                        yield Fact(entity=owner[1], attribute="text",
                                   value=el.text.strip()[:200], source=src)
                    elif tag == "SEMANTIC" and el.text:
                        yield Fact(entity=owner[1], attribute="semantic",
                                   value=el.text.strip(), source=src)
                if tag in (_DTC, _SVC, _DID) and stack and stack[-1][0] == tag:
                    stack.pop()
                el.clear()
        except Exception:
            return

    # ── A2L (ASAM MCD-2 MC) — measurement / calibration ──────────────────────
    _BEGIN = re.compile(r'/begin\s+(MEASUREMENT|CHARACTERISTIC|COMPU_METHOD)\s+(\S+)')

    def _a2l(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            return
        # A2L is brace-delimited /begin BLOCK name ... /end BLOCK. Parse each block.
        pos = 0
        for m in self._BEGIN.finditer(text):
            block, name = m.group(1), m.group(2)
            end = text.find(f"/end {block}", m.end())
            body = text[m.end():end if end > 0 else m.end() + 2000]
            kind = {"MEASUREMENT": "measurement", "CHARACTERISTIC": "characteristic",
                    "COMPU_METHOD": "compu_method"}[block]
            yield Fact(entity=name, attribute="autosar_type", value=kind, source=src)
            # the first quoted string after the name is the long description
            desc = re.search(r'"([^"]*)"', body)
            if desc and desc.group(1):
                yield Fact(entity=name, attribute="description",
                           value=desc.group(1)[:200], source=src)
            # ECU address (hex) common to MEASUREMENT/CHARACTERISTIC
            addr = re.search(r'\b(0x[0-9A-Fa-f]{4,})\b', body)
            if addr:
                yield Fact(entity=name, attribute="ecu_address",
                           value=addr.group(1), source=src)
            # unit (PHYS_UNIT "..." )
            unit = re.search(r'PHYS_UNIT\s+"([^"]*)"', body)
            if unit and unit.group(1):
                yield Fact(entity=name, attribute="unit", value=unit.group(1), source=src)
            # conversion method reference → relation
            conv = re.search(r'\b([A-Za-z_]\w*)\s*$', body.strip().split("\n")[0]) \
                if block != "COMPU_METHOD" else None
            cm = re.search(r'/begin\s+IF_DATA', body)  # placeholder guard
            # link MEASUREMENT/CHARACTERISTIC → its COMPU_METHOD (4th token convention)
            toks = body.split()
            for i, t in enumerate(toks):
                if t.startswith("CM_") or (t.isidentifier() and t.upper().startswith("COMPU")):
                    yield Fact(entity=name, attribute="uses_conversion",
                               value=t, source=src, kind="relation")
                    break

    def prose_chunks(self, path: str):
        return []
