#!/usr/bin/env python3
"""QueryPlan — explicit, deterministic query classification (v2 architecture).

The recurring entity-hijack bug came from scattered ad-hoc routing. The cure is an
EXPLICIT plan: classify a query ONCE into an intent + extract its parts, then let
the hybrid retriever gather candidates from the right channels and a single
reranker decide. No more "concept gate" / candidate-count / stopword patches.

Intent:
  FACT      — value of a named entity's attribute ("UFS for SKU1001")
  PROSE     — explanation / how-it-works / what-is-X ("how does the deserializer work")
  REVERSE   — find entities BY a value ("which parts use Winbond")
  MIXED     — both a named entity AND a concept ("specs and how the SerDes works")
  UNKNOWN_ENTITY — a precise id that isn't in the corpus → must return NOT_FOUND

The plan is data, not behavior; retrieve.py acts on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Explicit identifiers — a STRONG signal of a FACT query about a specific entity.
# Known part-number shapes PLUS any long numeric/alphanumeric token that looks
# like an id. ANY such token gets verified against the corpus → an unknown id
# (e.g. "9999999999") becomes UNKNOWN_ENTITY → NOT_FOUND, never a prose false hit.
_ID_RE = re.compile(r'\b(75\d{8}|86\d{8}|F0\d{2}[.\s]?\d{3}[.\s]?\d{3})\b')
_IDLIKE_RE = re.compile(r'\b(\d{7,}|[A-Z0-9]{2,}[-_]?\d{4,}|\d{3,}[-_][A-Z0-9]{2,})\b')

# Concept/topic words → the query wants understanding (PROSE), not a cell value.
_CONCEPT = re.compile(
    r'\b(how|why|explain|describe|work[s]?|using|pipeline|architecture|process|'
    r'interface|protocol|sequence|flow|handle[sd]?|overview|deserializer|serializer|'
    r'chip|wifi|wi-fi|bluetooth|driver|topology|register|module|design)\b', re.I)

# Attribute words → a precise spec lookup.
_ATTR = re.compile(
    r'\b(ufs|ram|memory|sxm|pcb|part\s*number|version|plant|variant|tuner|gnss|'
    r'ethernet|supplier|ecr|ppap|size|config|ip|mac|port|id|status)\b', re.I)

# Reverse: "which/what PARTS/ENTITIES ... use/have X" — must name a collection
# noun, else "what deserializer do we use" wrongly becomes a reverse-value lookup
# (it's a concept question). Require parts|entities|items|modules|boards|variants.
_REVERSE = re.compile(
    r'\b(which|what|list|find|all)\b.*\b(parts?|entities|items|modules?|boards?|'
    r'variants?|components?|files?)\b.*\b(use[sd]?|have|has|with|containing)\b', re.I)

# Relation/structural: "what calls X", "what does X call/import/depend on",
# "what is X in" — entity→entity edge questions (codegraph + AUTOSAR).
_RELATION = re.compile(r'\b(call[s]?|called by|import[s]?|depend[s]?|contain[s]?|'
                       r'reference[s]?|instantiat|inherit|implement|in (message|pdu|'
                       r'frame|bus)|on bus)\b', re.I)

_STOP = frozenset((
    "the", "is", "are", "was", "were", "of", "for", "and", "or", "in", "on", "to",
    "a", "an", "what", "which", "how", "does", "do", "did", "with", "has", "have",
    "had", "this", "that", "it", "its", "use", "used", "uses", "work", "works",
    "about", "from", "by", "as", "be", "can", "will", "would", "should", "we",
    "our", "i", "me", "my", "you", "get", "give", "show", "tell", "value", "price"))


@dataclass
class QueryPlan:
    intent: str                       # FACT|PROSE|REVERSE|MIXED|UNKNOWN_ENTITY
    entity: Optional[str] = None      # explicit id if present
    attr_terms: List[str] = field(default_factory=list)
    topic_terms: List[str] = field(default_factory=list)   # specific (non-stop) words
    raw: str = ""

    def summary(self) -> str:
        return (f"intent={self.intent} entity={self.entity} "
                f"attrs={self.attr_terms} topic={self.topic_terms}")


def plan_query(query: str, *, entity_exists=None) -> QueryPlan:
    """Build the plan. `entity_exists(id)->bool` lets us mark a precise id that is
    NOT in the corpus as UNKNOWN_ENTITY (→ NOT_FOUND, never unrelated data)."""
    q = query.strip()
    m = _ID_RE.search(q) or _IDLIKE_RE.search(q)
    explicit_id = m.group(0) if m else None
    has_concept = bool(_CONCEPT.search(q))
    has_attr = bool(_ATTR.search(q))
    is_reverse = bool(_REVERSE.search(q))
    is_relation = bool(_RELATION.search(q))
    topic = [w for w in re.findall(r'\w+', q.lower())
             if len(w) > 2 and w not in _STOP]

    # RELATION (structural/edge) query — "what calls X", "what is signal S in".
    # Highest priority when a relation verb + a target name are present.
    if is_relation and topic:
        return QueryPlan("RELATION", entity=explicit_id, topic_terms=topic, raw=q)

    # Explicit id present → FACT (or MIXED if it also asks a concept).
    if explicit_id:
        if entity_exists is not None and not entity_exists(explicit_id):
            return QueryPlan("UNKNOWN_ENTITY", entity=explicit_id, raw=q)
        intent = "MIXED" if has_concept else "FACT"
        return QueryPlan(intent, entity=explicit_id, attr_terms=_attrs(q),
                         topic_terms=topic, raw=q)

    if is_reverse:
        return QueryPlan("REVERSE", topic_terms=topic, attr_terms=_attrs(q), raw=q)

    # No explicit id. Concept words OR no attribute focus → PROSE. A bare attribute
    # term with no id is ambiguous → treat as PROSE/topic search (reranker decides).
    if has_concept or not has_attr:
        return QueryPlan("PROSE", topic_terms=topic, raw=q)
    return QueryPlan("PROSE", topic_terms=topic, attr_terms=_attrs(q), raw=q)


def _attrs(q: str) -> List[str]:
    return [m.group(0).lower() for m in _ATTR.finditer(q)]
