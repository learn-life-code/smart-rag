#!/usr/bin/env python3
"""Relations — entity→entity edges, expressed as Facts (the bridge to codegraph).

A relationship ("function A calls B", "signal S is in PDU P", "frame F on bus X")
is just a Fact whose VALUE is another entity and whose ATTRIBUTE names the edge.
This lets the SAME fact store + retrieval serve structural questions:

    Fact(entity="deserializer_init", attribute="calls", value="i2c_write",
         kind="relation", source="codegraph.db")

So Smart RAG ABSORBS codegraph's call-graph (and AUTOSAR's signal→PDU→frame→bus)
without a new store — the adapters emit relation Facts, and a RELATION query plan
walks them. Smart RAG does NOT replace codegraph's binary symbol/edge EXTRACTION
(its specialized parsers stay); it makes the result queryable with v2 quality.
"""
from __future__ import annotations

import re
from typing import List

from smart_rag.core.fact import Fact

# Attribute names that denote an entity→entity edge (vs a scalar value).
RELATION_ATTRS = {
    "calls", "called_by", "imports", "imported_by", "contains", "contained_in",
    "instantiates", "inherits", "implements", "references", "defined_in",
    "in_message", "in_pdu", "in_frame", "on_bus", "maps_to", "depends_on",
}

_VERB_TO_ATTR = [
    (re.compile(r'\bcall(s|ed)?\b', re.I), "calls"),
    (re.compile(r'\bimport(s|ed)?\b', re.I), "imports"),
    (re.compile(r'\bcontain(s|ed)?\b', re.I), "contains"),
    (re.compile(r'\bdepend(s|ent)?\b', re.I), "depends_on"),
    (re.compile(r'\bin (message|pdu|frame)\b', re.I), "in_message"),
    (re.compile(r'\bon bus\b', re.I), "on_bus"),
    (re.compile(r'\bdefine[sd]?\b', re.I), "defined_in"),
]


def make_relation(entity: str, attribute: str, target: str, *,
                  source: str = "", source_id: str = "", location: str = "") -> Fact:
    """Build a relation Fact (value = the target entity, kind='relation')."""
    return Fact(entity=entity, attribute=attribute, value=target, source=source,
                source_id=source_id, location=location, kind="relation")


def is_relation(fact: Fact) -> bool:
    return fact.kind == "relation" or fact.attribute.lower() in RELATION_ATTRS


def relation_attr_for(query: str) -> str:
    """Map a query verb to a relation attribute ('what calls X' → 'calls')."""
    for rx, attr in _VERB_TO_ATTR:
        if rx.search(query):
            return attr
    return ""


def walk_relations(store, entity: str, attribute: str = "",
                   reverse: bool = False, limit: int = 30,
                   internal_only: bool = False) -> List[dict]:
    """Return edges for an entity. reverse=True finds who points AT it
    (e.g. 'what CALLS X' = entities whose 'calls' value == X).

    internal_only=True keeps only edges whose CALLEE is itself a defined entity in
    the store (i.e. a function in YOUR code), dropping calls to libc/macros/externals
    — so 'what does X call' shows your own code's calls, not library noise. Each
    edge is tagged 'internal': True/False either way."""
    ents = set(store.entities)   # known symbols → an edge to one of these is internal
    out: List[dict] = []
    if reverse:
        for ent in store.entities:
            for attr, rows in store.lookup(ent).items():
                if attribute and attribute.lower() not in attr.lower():
                    continue
                if attr.lower() in RELATION_ATTRS:
                    for r in rows:
                        if r["value"] == entity:
                            out.append({"from": ent, "rel": attr, "to": entity,
                                        "source": (r["sources"] or ["?"])[0],
                                        "internal": True})  # caller is a known entity
                if len(out) >= limit:
                    return out
    else:
        for attr, rows in store.lookup(entity).items():
            if attribute and attribute.lower() not in attr.lower():
                continue
            if attr.lower() in RELATION_ATTRS:
                for r in rows:
                    is_internal = r["value"] in ents
                    if internal_only and not is_internal:
                        continue   # drop calls to externals (libc/macros)
                    out.append({"from": entity, "rel": attr, "to": r["value"],
                                "source": (r["sources"] or ["?"])[0],
                                "internal": is_internal})
            if len(out) >= limit:
                break
    return out
