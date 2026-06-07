"""Collectors — build a Smart RAG index over a SOURCE (folder, drive, PC, or a
live machine over SSH), so an AI agent can query it instead of grep+read.

The point: an agent burns most of its tokens reading files. Index a workspace ONCE
with a collector, then `answer(query)` returns cited facts in ~50 tokens instead of
the agent reading dozens of files. Indexes are named + persistent + incremental.

    from smart_rag.collectors import IndexManager
    mgr = IndexManager()
    mgr.build("myrepo", "C:/work/myrepo")          # index once (or refresh)
    print(mgr.answer("myrepo", "how does auth work").to_text())   # fast, cited
"""
from smart_rag.collectors.index import IndexManager
from smart_rag.collectors.fs import collect_fs

__all__ = ["IndexManager", "collect_fs"]

# ssh collector is optional (needs paramiko) — import lazily to avoid a hard dep.
try:
    from smart_rag.collectors.ssh import collect_ssh, collect_ssh_chunks  # noqa: F401
    __all__ += ["collect_ssh", "collect_ssh_chunks"]
except Exception:  # noqa: BLE001
    pass
