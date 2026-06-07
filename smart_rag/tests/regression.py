#!/usr/bin/env python3
"""Smart RAG regression suite — prove trustworthiness on REAL questions.

The review's key validation step: test actual questions (incl. rejection cases) so
we KNOW the system answers, abstains, and cites correctly — not just believe it.
Each case asserts the AnswerResult's status + that an expected value/source appears
(or, for rejection cases, that it ABSTAINS).

Run:
  py -3.13 -m smart_rag.tests.regression --matrix software/VCU_Variant_Matrix.xlsm
  py -3.13 -m smart_rag.tests.regression --folder F:/Test
Cases are tagged by which corpus they need; only runnable ones execute.
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from smart_rag import SmartRAG   # noqa: E402

# Each case: (query, expect_status_in, must_contain_or_None, why)
# must_contain: a substring that MUST appear in answer/evidence (None for rejection)
MATRIX_CASES = [
    ("what is the UFS size for SKU1001", {"ANSWERED", "PARTIAL"}, "128",
     "exact spec lookup"),
    ("RAM size for SKU1001", {"ANSWERED", "PARTIAL"}, "24", "exact spec lookup"),
    ("does SKU1001 have SXM", {"ANSWERED", "PARTIAL"}, None, "boolean attr present"),
    ("main PCB part number for SKU1001", {"ANSWERED", "PARTIAL"}, "PCB2150",
     "the PCB PN an old RAG got WRONG"),
    ("how much storage does SKU1001 have", {"ANSWERED", "PARTIAL"}, None,
     "INTENT: storage→UFS/RAM (no keyword)"),
    # rejection cases — MUST abstain
    ("what is the price of bitcoin", {"NOT_FOUND"}, None, "REJECT: unrelated"),
    ("how do I bake a cake", {"NOT_FOUND"}, None, "REJECT: unrelated"),
    ("UFS for 9999999999", {"NOT_FOUND", "PARTIAL"}, None, "REJECT/empty: unknown entity"),
]

FOLDER_CASES = [
    ("how does the deserializer work", {"ANSWERED"}, "deser", "concept from docs/datasheet"),
    ("audio routing", {"ANSWERED"}, None, "concept: audio docs"),
    ("MAX9296", {"ANSWERED", "PARTIAL", "NOT_FOUND"}, None, "part lookup across sources"),
    ("what is the price of bitcoin", {"NOT_FOUND"}, None, "REJECT: unrelated"),
]


def _run(d: SmartRAG, cases, label):
    print(f"\n=== {label} ({len(cases)} cases) ===")
    passed = 0
    for query, ok_status, must, why in cases:
        r = d.answer(query)
        text = (r.answer + " " + " ".join(e.text + " " + e.source for e in r.evidence)).lower()
        status_ok = r.status in ok_status
        contain_ok = (must is None) or (must.lower() in text)
        ok = status_ok and contain_ok
        passed += ok
        mark = "PASS" if ok else "FAIL"
        detail = f"status={r.status}"
        if not contain_ok:
            detail += f" (missing '{must}')"
        print(f"  [{mark}] {query[:42]:44} {detail:24} — {why}")
    print(f"  → {passed}/{len(cases)} passed")
    return passed, len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", help="path to VCU_Variant_Matrix.xlsm")
    ap.add_argument("--folder", help="path to a real engineering folder (e.g. F:/Test)")
    args = ap.parse_args()
    total_ok = total = 0

    if args.matrix and Path(args.matrix).exists():
        d = SmartRAG(); d.ingest(args.matrix, verbose=False)
        p, t = _run(d, MATRIX_CASES, f"matrix: {Path(args.matrix).name}")
        total_ok += p; total += t
    if args.folder and Path(args.folder).exists():
        d = SmartRAG(); d.ingest(args.folder, verbose=False)
        p, t = _run(d, FOLDER_CASES, f"folder: {args.folder}")
        total_ok += p; total += t

    if total:
        print(f"\n=== TOTAL: {total_ok}/{total} passed "
              f"({round(100*total_ok/total)}%) ===")
        sys.exit(0 if total_ok == total else 1)
    print("Nothing to run — pass --matrix and/or --folder.")


if __name__ == "__main__":
    main()
