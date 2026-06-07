#!/usr/bin/env python3
"""Config adapter — .ini/.cfg/.conf/.properties + .yaml/.yml/.toml → key=value FACTS.

entity = section / top-level key (or filename); attribute = key; value = value.
For nested YAML/TOML, the parent path is the entity and the leaf key the attribute
(e.g. database.host → entity=database, attribute=host). Simple, lossless, cited.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

_SECTION = re.compile(r'^\s*\[([^\]]+)\]\s*$')
_KV = re.compile(r'^\s*([^=:#;]+?)\s*[=:]\s*(.*?)\s*$')


class ConfigAdapter(Adapter):
    suffixes = (".ini", ".cfg", ".conf", ".properties", ".vsysvar",
                ".yaml", ".yml", ".toml")
    name = "config"

    def extract(self, path: str) -> Iterable[Fact]:
        low = path.lower()
        if low.endswith((".yaml", ".yml")):
            yield from self._structured(path, _load_yaml(path))
        elif low.endswith(".toml"):
            yield from self._structured(path, _load_toml(path))
        else:
            yield from self._ini(path)

    # ── ini / properties (line-based) ────────────────────────────────────────
    def _ini(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        section = src.rsplit(".", 1)[0]
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            return
        for line in lines:
            if not line.strip() or line.strip().startswith(("#", ";")):
                continue
            ms = _SECTION.match(line)
            if ms:
                section = ms.group(1).strip()
                continue
            mk = _KV.match(line)
            if mk:
                key, val = mk.group(1).strip(), mk.group(2).strip()
                if key and val:
                    yield Fact(entity=section, attribute=key, value=val, source=src)

    # ── yaml / toml (nested mapping) ─────────────────────────────────────────
    def _structured(self, path: str, data) -> Iterable[Fact]:
        src = os.path.basename(path)
        stem = src.rsplit(".", 1)[0]
        if not isinstance(data, dict):
            return
        # walk the tree; entity = the deepest mapping path, attribute = leaf key
        def walk(node, ent):
            for k, v in node.items():
                if isinstance(v, dict):
                    yield from walk(v, f"{ent}.{k}" if ent else str(k))
                elif isinstance(v, list):
                    # list of scalars → join; list of dicts → index as entities
                    if all(not isinstance(x, (dict, list)) for x in v):
                        if v:
                            yield Fact(entity=ent or stem, attribute=str(k),
                                       value=", ".join(str(x) for x in v), source=src)
                    else:
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                yield from walk(item, f"{ent}.{k}[{i}]" if ent else f"{k}[{i}]")
                elif v is not None and str(v).strip():
                    yield Fact(entity=ent or stem, attribute=str(k),
                               value=str(v), source=src)
        yield from walk(data, "")


def _load_yaml(path: str):
    try:
        import yaml  # PyYAML
        return yaml.safe_load(open(path, encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _load_toml(path: str):
    try:
        import tomllib  # Python 3.11+
        return tomllib.load(open(path, "rb"))
    except Exception:
        try:
            import tomli
            return tomli.load(open(path, "rb"))
        except Exception:
            return None
