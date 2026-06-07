#!/usr/bin/env python3
"""OpenAPI / Swagger adapter — API specs → endpoints, params, schemas as facts.

A standard most software teams have (OpenAPI 3.x / Swagger 2.0, .yaml or .json).
Turns "what endpoints touch auth" / "what does POST /users accept" into grounded,
cited answers.

  endpoint  → entity   "GET /users/{id}"
  attribute → summary, tags, auth, request_body, response_<code>
  relation  → operation --uses_schema--> Schema   (the $ref graph)

Schema-aware: only known OpenAPI structures are extracted; arbitrary YAML is not
flattened (it routes to the config adapter instead). Detection sniffs for the
`openapi:`/`swagger:` marker so it doesn't grab every yaml/json.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from smart_rag.adapters.base import Adapter
from smart_rag.core.fact import Fact

_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


class OpenAPIAdapter(Adapter):
    suffixes = ()   # detected by content, not just suffix (many .yaml/.json)
    name = "openapi"
    emits = ("endpoint", "schema")
    standard = "OpenAPI 3.x / Swagger 2.0"

    def can_handle(self, path: str) -> bool:
        low = path.lower()
        if not low.endswith((".yaml", ".yml", ".json")):
            return False
        try:
            head = open(path, encoding="utf-8", errors="replace").read(4000).lower()
            return ("openapi" in head or "swagger" in head) and "paths" in head
        except Exception:
            return False

    def _load(self, path: str):
        low = path.lower()
        try:
            if low.endswith(".json"):
                return json.load(open(path, encoding="utf-8", errors="replace"))
            import yaml
            return yaml.safe_load(open(path, encoding="utf-8", errors="replace"))
        except Exception:
            return None

    def extract(self, path: str) -> Iterable[Fact]:
        spec = self._load(path)
        if not isinstance(spec, dict):
            return
        src = os.path.basename(path)
        info = spec.get("info", {}) or {}
        api_name = info.get("title", src)
        if info.get("version"):
            yield Fact(entity=api_name, attribute="api_version",
                       value=str(info["version"]), source=src)

        for route, ops in (spec.get("paths") or {}).items():
            if not isinstance(ops, dict):
                continue
            for method, op in ops.items():
                if method.lower() not in _METHODS or not isinstance(op, dict):
                    continue
                ep = f"{method.upper()} {route}"
                yield Fact(entity=ep, attribute="autosar_type", value="endpoint",
                           source=src, kind="extracted")
                if op.get("summary"):
                    yield Fact(entity=ep, attribute="summary",
                               value=str(op["summary"])[:200], source=src)
                if op.get("operationId"):
                    yield Fact(entity=ep, attribute="operation_id",
                               value=str(op["operationId"]), source=src)
                for tag in (op.get("tags") or []):
                    yield Fact(entity=ep, attribute="tag", value=str(tag), source=src)
                if op.get("security"):
                    yield Fact(entity=ep, attribute="auth_required", value="true",
                               source=src)
                # parameters
                for p in (op.get("parameters") or []):
                    if isinstance(p, dict) and p.get("name"):
                        loc = p.get("in", "")
                        yield Fact(entity=ep, attribute=f"param_{loc}".rstrip("_"),
                                   value=str(p["name"]), source=src)
                # request body schema refs → relations
                for ref in _find_refs(op.get("requestBody")):
                    yield Fact(entity=ep, attribute="uses_schema", value=ref,
                               source=src, kind="relation")
                # responses
                for code, resp in (op.get("responses") or {}).items():
                    if isinstance(resp, dict) and resp.get("description"):
                        yield Fact(entity=ep, attribute=f"response_{code}",
                                   value=str(resp["description"])[:120], source=src)
                    for ref in _find_refs(resp):
                        yield Fact(entity=ep, attribute="returns_schema", value=ref,
                                   source=src, kind="relation")

        # schemas (components) as entities
        schemas = ((spec.get("components") or {}).get("schemas")
                   or spec.get("definitions") or {})
        for name, sch in schemas.items():
            yield Fact(entity=name, attribute="autosar_type", value="schema", source=src)
            props = (sch or {}).get("properties") or {}
            for prop, pdef in props.items():
                typ = (pdef or {}).get("type", "")
                yield Fact(entity=name, attribute=f"field:{prop}",
                           value=typ or "object", source=src)
            for req in (sch or {}).get("required", []):
                yield Fact(entity=name, attribute="required_field", value=str(req),
                           source=src)

    def prose_chunks(self, path: str):
        return []


def _find_refs(node, _depth: int = 0) -> Iterable[str]:
    """Yield the schema names referenced by $ref anywhere under node."""
    if _depth > 8 or node is None:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                yield v.rsplit("/", 1)[-1]
            else:
                yield from _find_refs(v, _depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _find_refs(item, _depth + 1)
