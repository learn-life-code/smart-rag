#!/usr/bin/env python3
"""Config adapter — .ini / .cfg / .conf / .properties → key=value FACTS.

entity = section ([RBSsettings]) or the filename; attribute = key; value = value.
So "what is DeviceName" → Device-A, with source. Simple, deterministic, lossless.
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
    suffixes = (".ini", ".cfg", ".conf", ".properties", ".vsysvar")
    name = "config"

    def extract(self, path: str) -> Iterable[Fact]:
        src = os.path.basename(path)
        section = src.rsplit(".", 1)[0]   # default entity = filename stem
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
