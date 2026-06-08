#!/usr/bin/env python3
"""Run ALL Smart RAG test suites — the CI gate (fails on any regression).

  py -3.13 -m smart_rag.tests.run_all              # unit + lifecycle (fast, no corpora)
  py -3.13 -m smart_rag.tests.run_all --full       # + adoption-gate eval (needs corpora)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def run(mod_args, label):
    print(f"\n########## {label} ##########")
    r = subprocess.run([sys.executable, "-m", *mod_args], cwd=str(ROOT))
    return r.returncode == 0


def main():
    full = "--full" in sys.argv
    ok = True
    ok &= run(["smart_rag.tests.test_adapters"], "per-adapter unit/golden")
    ok &= run(["smart_rag.tests.test_lifecycle"], "source lifecycle")
    ok &= run(["smart_rag.tests.test_adversarial"], "adversarial (review-found cases)")
    ok &= run(["smart_rag.tests.test_collectors"], "collectors (fs/index/ssh-guard)")
    ok &= run(["smart_rag.tests.test_plugin"], "consumer evidence plug-in")
    if full:
        matrix = ROOT / "software" / "VCU_Variant_Matrix.xlsm"
        if matrix.exists():
            ok &= run(["scripts.eval", "--matrix", str(matrix)], "adoption gate (matrix)")
    print("\n==================================")
    print("ALL SUITES: " + ("PASS" if ok else "FAIL (regression)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
