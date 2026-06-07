#!/usr/bin/env python3
"""Per-adapter unit + golden tests — deterministic, fast, CI-able.

Each adapter is tested on a tiny in-repo golden fixture: ingest → assert the
expected facts/prose/relations appear (and that empty/garbage is handled). No
network, no big corpora. This is the review's "unit tests for every adapter +
golden tests + CI that fails on regression".

Run: py -3.13 -m smart_rag.tests.test_adapters
"""
import os
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_PASS = _FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    _PASS += bool(cond); _FAIL += (not cond)


def _w(d, name, body):
    p = os.path.join(d, name)
    open(p, "w", encoding="utf-8").write(body)
    return p


def main():
    from smart_rag.adapters import adapter_for
    tmp = tempfile.mkdtemp(prefix="distill_ad_")

    # ── CSV (tabular) ────────────────────────────────────────────────────────
    f = _w(tmp, "parts.csv", "id,UFS,RAM\nSKU1001,128,24\nSKU1002,256,32\n")
    facts = list(adapter_for(f).extract(f))
    vals = {(x.entity, x.attribute): x.value for x in facts}
    check("csv: entity SKU1001 UFS=128", vals.get(("SKU1001", "UFS")) == "128")
    check("csv: keeps a second row", ("SKU1002", "RAM") in vals)

    # ── JSON ─────────────────────────────────────────────────────────────────
    f = _w(tmp, "rec.json", '[{"id":"X1","color":"red"},{"id":"X2","color":"blue"}]')
    facts = list(adapter_for(f).extract(f))
    check("json: X1 color=red", any(x.entity == "X1" and x.value == "red" for x in facts))

    # ── INI (config) ─────────────────────────────────────────────────────────
    f = _w(tmp, "cfg.ini", "[net]\nIP=10.0.0.1\nName=Device-A\n")
    facts = list(adapter_for(f).extract(f))
    check("ini: net.IP=10.0.0.1", any(x.entity == "net" and x.attribute == "IP"
                                      and x.value == "10.0.0.1" for x in facts))

    # ── DBC (autosar relations) ──────────────────────────────────────────────
    f = _w(tmp, "bus.dbc",
           "BO_ 256 EngineData: 8 ECM\n SG_ Speed : 0|16@1+ (1,0) [0|9] \"rpm\" TCM\n")
    facts = list(adapter_for(f).extract(f))
    rels = [x for x in facts if x.kind == "relation"]
    check("dbc: Speed in_message EngineData (relation)",
          any(x.entity == "Speed" and x.attribute == "in_message"
              and x.value == "EngineData" for x in rels))
    check("dbc: EngineData contains Speed (reverse relation)",
          any(x.entity == "EngineData" and x.attribute == "contains"
              and x.value == "Speed" for x in rels))

    # ── logs (component → events) ────────────────────────────────────────────
    f = _w(tmp, "run.log",
           "12-01 10:00:00.000 1 1 E QSYM/AUD: audio mute failed errno=2\n"
           "12-01 10:00:01.000 1 1 I QSYM/AUD: started ok\n")
    facts = list(adapter_for(f).extract(f))
    check("logs: only NOTABLE lines kept (error, not 'started ok')",
          any("mute failed" in x.value for x in facts)
          and not any("started ok" in x.value for x in facts))

    # ── docs (markdown prose) ────────────────────────────────────────────────
    f = _w(tmp, "notes.md", "# Deserializer\nThe SER-100 remaps VC-IDs.\n")
    pr = list(adapter_for(f).prose_chunks(f))
    check("md: prose chunk captured", any("SER-100" in c["text"] for c in pr))

    # ── code (symbol facts + prose) ──────────────────────────────────────────
    f = _w(tmp, "x.py", "# helper\ndef deserialize(b):\n    return b\n")
    facts = list(adapter_for(f).extract(f))
    check("code: symbol 'deserialize' defined_in",
          any(x.entity == "deserialize" and x.attribute == "defined_in" for x in facts))

    # ── call graph (Python AST): A calls B edges + noise filtered ────────────
    f = _w(tmp, "cg.py",
           "def helper(x):\n    return x + 1\n\n"
           "def main():\n    print('hi')\n    return helper(5)\n")
    facts = list(adapter_for(f).extract(f))
    edges = [x for x in facts if x.attribute == "calls" and x.kind == "relation"]
    check("callgraph: 'main calls helper' edge present",
          any(x.entity == "main" and x.value == "helper" for x in edges))
    check("callgraph: builtin 'print' filtered as noise",
          not any(x.value == "print" for x in edges))

    # ── empty / garbage honesty ──────────────────────────────────────────────
    f = _w(tmp, "blank.ini", "\n; nothing\n")
    check("empty ini → 0 facts", len(list(adapter_for(f).extract(f))) == 0)

    # ── tabular emit (improved TOON: schema-once + types + provenance) ───────
    from smart_rag.core.tabular_emit import emit_tabular
    out = emit_tabular(
        [{"entity": "P1", "UFS": "128GB", "SXM": "yes", "source": "s.csv"}],
        with_source=True)
    check("tabular emit: typed header (UFS:num/gb) + provenance (| src)",
          "UFS:num/gb" in out and "| src" in out and "128" in out and "128GB" not in out)
    check("tabular emit: bool canonicalized (yes→true)", "true" in out)

    # ── world-standard adapters (YAML, SPICE, OpenAPI, ODX, A2L) ─────────────
    f = _w(tmp, "c.yaml", "db:\n  host: h1\n  port: 5432\n")
    facts = list(adapter_for(f).extract(f))
    check("yaml: nested db.host=h1", any(x.entity == "db" and x.attribute == "host"
                                         and x.value == "h1" for x in facts))

    f = _w(tmp, "ckt.sp", "* c\nR1 a b 1k\nM1 d g s s NMOS\n")
    facts = list(adapter_for(f).extract(f))
    rels = [x for x in facts if x.kind == "relation"]
    check("spice: M1 mosfet connected_to its nets (relation graph)",
          any(x.entity == "M1" and x.attribute == "connected_to" for x in rels))

    f = _w(tmp, "api.yaml",
           "openapi: 3.0.0\ninfo:\n  title: API\n  version: '1'\npaths:\n"
           "  /u:\n    get:\n      summary: list\n")
    facts = list(adapter_for(f).extract(f))
    check("openapi: endpoint 'GET /u' extracted",
          any(x.entity == "GET /u" and x.value == "endpoint" for x in facts))

    f = _w(tmp, "d.odx", '<?xml version="1.0"?><ODX><DTC><SHORT-NAME>P0420'
           '</SHORT-NAME><TROUBLE-CODE>1056</TROUBLE-CODE></DTC></ODX>')
    facts = list(adapter_for(f).extract(f))
    check("odx: DTC P0420 with trouble_code (ISO 22901)",
          any(x.entity == "P0420" and x.attribute == "trouble_code" for x in facts))

    f = _w(tmp, "e.a2l", '/begin MEASUREMENT Speed "RPM" UWORD CM_X 0 0 0 9\n'
           '  PHYS_UNIT "rpm"\n/end MEASUREMENT')
    facts = list(adapter_for(f).extract(f))
    check("a2l: MEASUREMENT Speed with unit (ASAM MCD-2 MC)",
          any(x.entity == "Speed" and x.attribute == "unit" and x.value == "rpm"
              for x in facts))

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== adapters: {_PASS}/{_PASS+_FAIL} passed ===")
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
