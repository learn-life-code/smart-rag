#!/usr/bin/env python3
"""Call-graph extraction — real 'A calls B' edges from SOURCE, for the code adapter.

The gap codegraph (binary-focused) and plain symbol regex both have: a true call
graph from source. `call_edges(text, src, ext)` yields RELATION facts:

  A calls B   → Fact(entity=A, attribute="calls", value=B, kind="relation")
  M.method    → Fact(entity=method, attribute="in_class", value=M, kind="relation")
  inherits    → Fact(entity=Class, attribute="inherits", value=Base, kind="relation")

Backends (best available, graceful):
  * Python      → stdlib `ast` (exact, zero deps) — always on.
  * Other langs → tree-sitter (optional). If tree-sitter + the grammar aren't
                  installed, non-Python files simply get no call edges (symbols +
                  prose from the code adapter still work). No hard dependency.
"""
from __future__ import annotations

import ast
from typing import Iterable

from smart_rag.core.fact import Fact

# Builtins + ubiquitous methods that are NOISE in a call graph (you want
# 'A calls my_helper', not 'A calls append').
_NOISE = frozenset((
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "range", "enumerate", "zip", "sorted", "reversed", "min", "max", "sum", "abs",
    "any", "all", "map", "filter", "open", "isinstance", "type", "append", "extend",
    "insert", "pop", "get", "keys", "values", "items", "setdefault", "update", "add",
    "remove", "join", "split", "strip", "lower", "upper", "replace", "format",
    "startswith", "endswith", "find", "index", "super", "getattr", "setattr",
    "hasattr", "next", "iter", "round", "sort", "lstrip", "rstrip", "encode",
    "decode", "read", "write", "close", "group", "match", "search", "compile"))


def call_edges(text: str, src: str, ext: str) -> Iterable[Fact]:
    """Yield call-graph RELATION facts for a source file (by language)."""
    if ext == ".py":
        yield from _python_edges(text, src)
    else:
        yield from _treesitter_edges(text, src, ext)


# ── Python: stdlib ast (exact) ───────────────────────────────────────────────
def _python_edges(text: str, src: str) -> Iterable[Fact]:
    try:
        tree = ast.parse(text)
    except Exception:
        return
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    # map function node → enclosing class (for in_class) in one pass
    enclosing = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosing[item] = cls.name
        for base in cls.bases:
            b = _name(base)
            if b:
                yield Fact(entity=cls.name, attribute="inherits", value=b,
                           source=src, kind="relation")
    for fn in (n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        if fn in enclosing:
            yield Fact(entity=fn.name, attribute="in_class", value=enclosing[fn],
                       source=src, kind="relation")
        seen = set()
        for c in (x for x in ast.walk(fn) if isinstance(x, ast.Call)):
            callee = _name(c.func)
            if not callee or callee in _NOISE or callee in seen:
                continue
            seen.add(callee)
            yield Fact(entity=fn.name, attribute="calls", value=callee, source=src,
                       kind="relation",
                       confidence=1.0 if callee in defined else 0.6)


def _name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


# ── Other languages: tree-sitter (optional) ──────────────────────────────────
_TS_CACHE = {}   # ext -> (parser, query) or None
_TS_LANG = {".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
            ".java": "java", ".js": "javascript", ".ts": "typescript",
            ".go": "go", ".rs": "rust", ".rb": "ruby"}


def _treesitter_edges(text: str, src: str, ext: str) -> Iterable[Fact]:
    lang_name = _TS_LANG.get(ext)
    if not lang_name:
        return
    parser = _ts_parser(lang_name)
    if parser is None:
        return   # tree-sitter not installed → no call edges (graceful)
    try:
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        return
    # Walk the AST: function_definition nodes → their identifier; call_expression
    # nodes inside → callee. Tree-sitter node type names are reasonably consistent.
    yield from _ts_walk(tree.root_node, text, src)


def _ts_parser(lang_name: str):
    if lang_name in _TS_CACHE:
        return _TS_CACHE[lang_name]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang_name)
    except Exception:
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser(lang_name)
        except Exception:
            parser = None
    _TS_CACHE[lang_name] = parser
    return parser


_FUNC_NODES = {"function_definition", "function_declaration", "method_definition",
               "method_declaration", "function_item", "constructor_declaration"}


def _ts_walk(node, text: str, src: str, _fn=None) -> Iterable[Fact]:
    t = node.type
    if t in _FUNC_NODES:
        name = _ts_child_name(node, text)
        if name:
            _fn = name
    if t in ("call_expression", "method_invocation", "call") and _fn:
        callee = _ts_callee(node, text)
        if callee and callee not in _NOISE:
            yield Fact(entity=_fn, attribute="calls", value=callee, source=src,
                       kind="relation", confidence=0.6)
    for child in node.children:
        yield from _ts_walk(child, text, src, _fn)


def _ts_child_name(node, text: str) -> str:
    for c in node.children:
        if c.type in ("identifier", "field_identifier", "name"):
            return text[c.start_byte:c.end_byte]
    return ""


def _ts_callee(node, text: str) -> str:
    # the function being called is usually the first identifier-ish child
    for c in node.children:
        if c.type in ("identifier", "field_identifier"):
            return text[c.start_byte:c.end_byte]
        if c.type in ("attribute", "field_expression", "selector_expression"):
            return _ts_child_name(c, text) or text[c.start_byte:c.end_byte].split(".")[-1]
    return ""
