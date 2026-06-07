#!/usr/bin/env python3
"""AUTOSAR / CAN adapter — .arxml / .dbc / .xml → signal & message FACTS.

Vehicle config lives here: which signals/messages exist, their bus, length, init
values. Turns "audio signal", "display message" into queryable structured facts.
  * .dbc  — CAN database: BO_ (message) / SG_ (signal) line format.
  * .arxml/.xml — AUTOSAR: SHORT-NAME elements + their child values.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

_BO = re.compile(r'^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)')        # message
_SG = re.compile(r'^\s*SG_\s+(\w+)\s*:\s*([\d|@+\-]+)\s*\(([^)]*)\)\s*\[([^\]]*)\]\s*"([^"]*)"\s*(\w+)?')  # signal


class AutosarAdapter(Adapter):
    suffixes = (".dbc", ".arxml", ".xml")
    name = "autosar"

    def extract(self, path: str) -> Iterable[Fact]:
        low = path.lower()
        if low.endswith(".dbc"):
            yield from self._dbc(path)
        else:
            yield from self._arxml(path)

    # ── CAN .dbc ─────────────────────────────────────────────────────────────
    def _dbc(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        cur_msg = None
        for line in open(path, encoding="utf-8", errors="replace"):
            mb = _BO.match(line)
            if mb:
                cur_msg = mb.group(2)
                yield Fact(entity=cur_msg, attribute="kind", value="CAN message", source=src)
                yield Fact(entity=cur_msg, attribute="CAN id", value=mb.group(1), source=src)
                yield Fact(entity=cur_msg, attribute="length (bytes)", value=mb.group(3), source=src)
                # transmitter = a relation (message → ECU/node that sends it)
                yield Fact(entity=cur_msg, attribute="on_bus", value=mb.group(4),
                           source=src, kind="relation")
                continue
            ms = _SG.match(line)
            if ms and cur_msg:
                sig = ms.group(1)
                yield Fact(entity=sig, attribute="kind", value="CAN signal", source=src)
                # RELATION (schema-aware, not flattened): signal → message, and the
                # reverse message → signal, so "what signals are in message M" and
                # "what message is signal S in" both work (the review's hierarchy req).
                yield Fact(entity=sig, attribute="in_message", value=cur_msg,
                           source=src, kind="relation")
                yield Fact(entity=cur_msg, attribute="contains", value=sig,
                           source=src, kind="relation")
                if ms.group(5):
                    yield Fact(entity=sig, attribute="unit", value=ms.group(5), source=src)
                if ms.group(6):
                    yield Fact(entity=sig, attribute="receiver", value=ms.group(6),
                               source=src, kind="relation")

    # AUTOSAR element types we explicitly support (schema-aware, not arbitrary XML).
    _SUPPORTED = {
        "I-SIGNAL", "SYSTEM-SIGNAL", "I-SIGNAL-I-PDU", "I-PDU", "PDU",
        "CAN-FRAME", "FRAME", "ETHERNET-FRAME", "CAN-CLUSTER", "ECU-INSTANCE",
        "I-SIGNAL-TO-I-PDU-MAPPING", "PDU-TO-FRAME-MAPPING",
        "DATA-TRANSFORMATION", "COMPU-METHOD", "DIAGNOSTIC-DATA-IDENTIFIER",
    }

    # ── AUTOSAR .arxml/.xml — SCHEMA-AWARE (preserve hierarchy + *-REF edges) ──
    def _arxml(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        try:
            import xml.etree.ElementTree as ET
        except Exception:
            return
        try:
            # ELEMENT-CONTEXT STACK: each supported container on the stack carries
            # its OWN identity. We only emit facts/refs when INSIDE a supported
            # object (stack non-empty) and attribute them to the NEAREST supported
            # owner (stack[-1]). This stops the global-cur_name drift + the flood
            # of generic definition/value facts from unsupported containers.
            it = ET.iterparse(path, events=("start", "end"))
            stack = []          # list of [type, name|None] for OPEN supported elems
            _ALLOWED_ATTRS = {"CATEGORY", "LENGTH", "INIT-VALUE", "LOWER-LIMIT",
                              "UPPER-LIMIT"}
            for ev, el in it:
                tag = el.tag.split("}")[-1]
                if ev == "start":
                    if tag in self._SUPPORTED:
                        stack.append([tag, None])   # name filled on its SHORT-NAME
                    continue
                # ev == "end"
                owner = stack[-1] if stack else None
                if tag == "SHORT-NAME" and el.text and owner is not None and owner[1] is None:
                    owner[1] = el.text.strip()      # name THIS supported object
                    yield Fact(entity=owner[1], attribute="autosar_type",
                               value=owner[0], source=src)
                    if len(stack) >= 2 and stack[-2][1]:   # nested under another object
                        parent = stack[-2]
                        rel = ("in_pdu" if "PDU" in parent[0] else
                               "in_frame" if "FRAME" in parent[0] else "in_cluster")
                        yield Fact(entity=owner[1], attribute=rel, value=parent[1],
                                   source=src, kind="relation")
                elif tag.endswith("-REF") and el.text and owner is not None and owner[1]:
                    target = el.text.strip().rsplit("/", 1)[-1]
                    if target and len(target) <= 80:
                        rel = tag.replace("-REF", "").replace("-", "_").lower() or "references"
                        yield Fact(entity=owner[1], attribute=rel, value=target,
                                   source=src, kind="relation")
                elif (owner is not None and owner[1] and el.text and el.text.strip()
                      and tag in _ALLOWED_ATTRS):
                    val = el.text.strip()
                    if 0 < len(val) <= 80:
                        yield Fact(entity=owner[1], attribute=tag.replace("-", " ").title(),
                                   value=val, source=src)
                if tag in self._SUPPORTED and stack and stack[-1][0] == tag:
                    stack.pop()
                el.clear()
        except Exception:  # noqa: BLE001
            # Malformed/non-AUTOSAR XML → yield nothing; the ingest coverage records
            # this source as empty/error honestly. No need to spam stdout.
            return
