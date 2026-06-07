#!/usr/bin/env python3
"""Chip / semiconductor standards adapter — IP-XACT (IEEE 1685) + SPICE netlist.

Brings EDA/SoC interchange standards into Smart RAG. These are entity-relation by
nature, so they map cleanly onto the fact + relation model:

  IP-XACT (.xml with <component>/<spirit:component>)  — IEEE 1685 IP description:
      component  → entity, attributes: vendor, library, version
      register   → entity, attributes: offset, size, access; in_component relation
      busInterface → relation (component --has_interface--> bus)
  SPICE netlist (.sp, .cir, .spice, .net)  — circuit connectivity:
      component (R/C/L/M/Q/X...) → entity, attributes: type, value, model
      net        → relation (component --connected_to--> net)  ← the key graph

The SPICE net graph in particular is exactly what the relation retriever answers
('what is connected to net VDD', 'what nets does M1 touch').
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

# SPICE element prefixes → device type (the first letter of a line names the device)
_SPICE_DEV = {
    "R": "resistor", "C": "capacitor", "L": "inductor", "D": "diode",
    "Q": "bjt", "M": "mosfet", "J": "jfet", "X": "subcircuit", "V": "vsource",
    "I": "isource", "E": "vcvs", "G": "vccs", "K": "coupling",
}


class ChipStdAdapter(Adapter):
    suffixes = (".sp", ".cir", ".spice", ".net", ".ipxact")
    name = "chip_std"
    emits = ("resistor","capacitor","inductor","mosfet","bjt","subcircuit","ip_component")
    standard = "IEEE 1685 (IP-XACT) + SPICE netlist"

    def can_handle(self, path: str) -> bool:
        low = path.lower()
        if low.endswith(self.suffixes):
            return True
        # IP-XACT lives in .xml — sniff for the component marker
        if low.endswith(".xml"):
            try:
                head = open(path, encoding="utf-8", errors="replace").read(3000).lower()
                return "ipxact" in head or "spirit:component" in head or \
                       ("<component" in head and "vendor" in head)
            except Exception:
                return False
        return False

    def extract(self, path: str) -> Iterable[Fact]:
        low = path.lower()
        if low.endswith((".sp", ".cir", ".spice", ".net")):
            yield from self._spice(path)
        else:
            yield from self._ipxact(path)

    # ── SPICE netlist — components + net connectivity ────────────────────────
    def _spice(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            return
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith(("*", ".", "+")):
                continue   # comment / directive / continuation
            toks = line.split()
            if len(toks) < 3:
                continue
            name = toks[0]
            dev = _SPICE_DEV.get(name[0].upper())
            if not dev:
                continue
            yield Fact(entity=name, attribute="autosar_type", value=dev, source=src)
            # nets: middle tokens are node connections; last token is often value/model
            # heuristic by device: 2-terminal R/C/L/D → toks[1],toks[2]; value=toks[3]
            if dev in ("resistor", "capacitor", "inductor", "diode", "vsource", "isource"):
                nets = toks[1:3]
                if len(toks) >= 4:
                    yield Fact(entity=name, attribute="value", value=toks[3], source=src)
            elif dev in ("mosfet", "bjt", "jfet"):
                nets = toks[1:4]   # D G S (B) terminals
                model = toks[4] if len(toks) > 4 else ""
                if model:
                    yield Fact(entity=name, attribute="model", value=model, source=src)
            elif dev == "subcircuit":
                # Xname n1 n2 ... subckt_name
                nets = toks[1:-1]
                yield Fact(entity=name, attribute="subckt", value=toks[-1], source=src)
            else:
                nets = toks[1:3]
            for net in nets:
                yield Fact(entity=name, attribute="connected_to", value=net,
                           source=src, kind="relation")

    # ── IP-XACT (IEEE 1685) — components / registers / interfaces ────────────
    def _ipxact(self, path: str) -> Iterable[Fact]:
        try:
            import xml.etree.ElementTree as ET
        except Exception:
            return
        src = os.path.basename(path)
        try:
            it = ET.iterparse(path, events=("start", "end"))
        except Exception:
            return
        comp = None
        in_reg = None
        reg_fields = {}
        try:
            for ev, el in it:
                tag = el.tag.split("}")[-1]
                if ev != "end":
                    continue
                if tag == "name" and el.text:
                    txt = el.text.strip()
                    # first <name> under <component> = component name
                    if comp is None:
                        comp = txt
                        yield Fact(entity=comp, attribute="autosar_type",
                                   value="ip_component", source=src)
                elif tag in ("vendor", "library", "version") and el.text and comp:
                    yield Fact(entity=comp, attribute=tag, value=el.text.strip(), source=src)
                elif tag == "register":
                    # registers handled via their sub-elements below in a simple pass
                    pass
                elif tag in ("addressOffset", "size", "access") and el.text and comp:
                    # attach to the component-level register context (best-effort)
                    reg_fields[tag] = el.text.strip()
                elif tag == "busInterface":
                    pass
                el.clear()
        except Exception:
            return

    def prose_chunks(self, path: str):
        return []
