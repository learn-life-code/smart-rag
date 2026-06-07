#!/usr/bin/env python3
"""Collector tests — fs walk/skip, named index build+query+reopen, SSH write-guard.

Run: python -m smart_rag.tests.test_collectors
"""
import os
import shutil
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


def main():
    from smart_rag.collectors.fs import collect_fs
    from smart_rag.collectors.ssh import _is_read_only
    from smart_rag.collectors import IndexManager

    tmp = tempfile.mkdtemp(prefix="sr_coll_")

    # ── fs collector: walks files, skips noise dirs + binary exts ────────────
    os.makedirs(os.path.join(tmp, "src"))
    os.makedirs(os.path.join(tmp, "node_modules", "x"))
    os.makedirs(os.path.join(tmp, ".git"))
    open(os.path.join(tmp, "src", "a.py"), "w").write("def f(): pass")
    open(os.path.join(tmp, "node_modules", "x", "b.js"), "w").write("noise")
    open(os.path.join(tmp, ".git", "config"), "w").write("noise")
    open(os.path.join(tmp, "img.png"), "wb").write(b"\x89PNG")
    found = [os.path.basename(p) for p in collect_fs(tmp)]
    check("fs: indexes real source file (a.py)", "a.py" in found)
    check("fs: skips node_modules", "b.js" not in found)
    check("fs: skips .git", "config" not in found)
    check("fs: skips binary (png)", "img.png" not in found)

    # ── ssh write-guard: read-only allowed, destructive refused ──────────────
    check("ssh: read-only 'cat /etc/os-release' allowed", _is_read_only("cat /etc/os-release"))
    check("ssh: 'ps -e' allowed", _is_read_only("ps -e"))
    check("ssh: REFUSES 'rm -rf /'", not _is_read_only("rm -rf /"))
    check("ssh: REFUSES 'systemctl stop x'", not _is_read_only("systemctl stop x"))
    check("ssh: REFUSES redirect '> /etc/x'", not _is_read_only("echo x > /etc/x"))
    check("ssh: REFUSES reboot", not _is_read_only("reboot"))

    # ── named index: build → query → reopen (persisted) ─────────────────────
    home = os.path.join(tmp, "home")
    os.makedirs(os.path.join(tmp, "proj"))
    open(os.path.join(tmp, "proj", "notes.md"), "w").write(
        "# Auth\nThe login service validates the token before granting access.")
    mgr = IndexManager(home=home)
    mgr.build("proj", os.path.join(tmp, "proj"), verbose=False)
    r = mgr.answer("proj", "how does login validate")
    check("index: build+query answers from indexed folder",
          r.status in ("ANSWERED", "PARTIAL"))
    check("index: catalog lists the index", "proj" in mgr.list())
    # reopen (fresh manager) = persisted, no rebuild
    mgr2 = IndexManager(home=home)
    r2 = mgr2.answer("proj", "login service token")
    check("index: reopen queries persisted store (no rebuild)",
          r2.status in ("ANSWERED", "PARTIAL"))
    check("index: unknown query abstains", mgr2.answer("proj", "price of bitcoin").status
          in ("NOT_FOUND", "INSUFFICIENT_EVIDENCE"))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== collectors: {_PASS}/{_PASS+_FAIL} passed ===")
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
