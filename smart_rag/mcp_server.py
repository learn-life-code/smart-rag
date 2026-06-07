#!/usr/bin/env python3
"""Smart RAG MCP server — the agent front door.

Exposes Smart RAG as MCP tools so an AI agent (Claude Code, Cursor, etc.) queries a
pre-built index INSTEAD of grep+read across a workspace. The agent asks one
question and gets cited facts in ~50 tokens, rather than reading dozens of files.

Tools:
  smartrag_index(name, path)   — build/refresh an index over a folder (once)
  smartrag_answer(name, query) — cited, abstaining answer (the fast path)
  smartrag_search(name, entity, attribute) — exact fact lookup, no LLM
  smartrag_list()              — what indexes exist

Run:  python -m smart_rag.mcp_server
Register in your MCP client (e.g. Claude Code .mcp.json):
  {"mcpServers": {"smartrag": {"command": "python", "args": ["-m", "smart_rag.mcp_server"]}}}

Falls back to a plain stdio JSON loop if the `mcp` package isn't installed, so it
works without extra deps (the agent can still call it).
"""
from __future__ import annotations

import json
import os
import sys

# Self-locate the package so the server runs whether launched as
#   python -m smart_rag.mcp_server   (needs smart_rag on the path), OR
#   python /abs/path/to/smart_rag/mcp_server.py   (path-independent — robust for MCP
#   clients that don't reliably honor a PYTHONPATH env, e.g. Claude on Windows).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smart_rag.collectors import IndexManager  # noqa: E402

_MGR = IndexManager()


# ── tool implementations (shared by MCP + fallback) ──────────────────────────
def _index(name: str, path: str) -> str:
    info = _MGR.build(name, path, verbose=False)
    msg = (f"Indexed '{name}': {info.get('ingested')} files, "
           f"{info.get('entities')} entities, {info.get('facts')} facts "
           f"in {info.get('build_secs')}s. Query with smartrag_answer('{name}', ...).")
    if info.get("codegraph_advice"):
        msg += f"\n\n{info['codegraph_advice']}"
    return msg


def _answer(name: str, query: str) -> str:
    return _MGR.answer(name, query).to_text()


def _search(name: str, entity: str = "", attribute: str = "") -> str:
    hits = _MGR.search(name, entity=entity or None, attribute=attribute or None, limit=30)
    if not hits:
        return "No matching facts."
    return "\n".join(f"{h.entity} · {h.attribute}: {h.value}  [{h.source}]" for h in hits)


def _list() -> str:
    cat = _MGR.list()
    if not cat:
        return "No indexes yet. Build one with smartrag_index(name, path)."
    return "\n".join(f"{n}: {m.get('entities')} entities from {m.get('source')}"
                     for n, m in cat.items())


# ── MCP server (preferred) ───────────────────────────────────────────────────
def _run_mcp() -> bool:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        return False
    app = FastMCP("smartrag")

    @app.tool()
    def smartrag_index(name: str, path: str) -> str:
        """Build or refresh a Smart RAG index over a folder. Run once per workspace;
        then query it fast. Incremental — re-running after edits is cheap."""
        return _index(name, path)

    @app.tool()
    def smartrag_answer(name: str, query: str) -> str:
        """Answer a natural-language question from an indexed workspace, with cited
        sources. Returns NOT_FOUND honestly if the answer isn't there. Use this
        INSTEAD of grep+read to save tokens and get grounded results."""
        return _answer(name, query)

    @app.tool()
    def smartrag_search(name: str, entity: str = "", attribute: str = "") -> str:
        """Exact fact lookup in an index (no LLM). e.g. entity='UserService'."""
        return _search(name, entity, attribute)

    @app.tool()
    def smartrag_list() -> str:
        """List the available Smart RAG indexes."""
        return _list()

    app.run()
    return True


# ── stdio JSON fallback (no mcp package) ─────────────────────────────────────
def _run_stdio() -> None:
    """Minimal line-delimited JSON-RPC-ish loop: {"tool": "...", "args": {...}}."""
    sys.stderr.write("[smartrag] mcp package not found — stdio JSON fallback. "
                     'Send: {"tool":"answer","args":{"name":"x","query":"..."}}\n')
    sys.stderr.flush()
    dispatch = {"index": lambda a: _index(a["name"], a["path"]),
                "answer": lambda a: _answer(a["name"], a["query"]),
                "search": lambda a: _search(a.get("name", ""), a.get("entity", ""),
                                            a.get("attribute", "")),
                "list": lambda a: _list()}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            fn = dispatch.get(req.get("tool", ""))
            out = fn(req.get("args", {})) if fn else f"unknown tool: {req.get('tool')}"
            print(json.dumps({"result": out}))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": str(e)}))
        sys.stdout.flush()


def main() -> None:
    if not _run_mcp():
        _run_stdio()


if __name__ == "__main__":
    main()
