#!/usr/bin/env python3
"""AnswerResult — a structured, trustworthy answer (raw evidence first).

The review's key point: a plain-text answer hides whether it's confident, what it's
grounded on, what's missing, and whether versions conflict. AnswerResult makes all
of that explicit so a user (or a downstream LLM) can decide whether to trust it.

  status      ANSWERED | PARTIAL | CONFLICT | NOT_FOUND
  answer      the grounded answer text (values verbatim from sources)
  evidence    list of {text, source, version?} actually used — the receipts
  confidence  HIGH | MEDIUM | LOW (from match strength / coverage)
  missing     attributes/info the user asked for but isn't recorded
  conflicts   attributes with >1 distinct value across versions (flag, don't guess)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Evidence:
    text: str
    source: str = ""
    version: str = ""
    score: Optional[float] = None
    location: str = ""            # precise locator (cell/line/page) for audit
    kind: str = "extracted"       # "extracted" (verbatim) vs "inferred" (derived)


@dataclass
class AnswerResult:
    status: str                       # ANSWERED|PARTIAL|CONFLICT|NOT_FOUND
    answer: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    confidence: str = "LOW"           # HIGH|MEDIUM|LOW
    missing: List[str] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)  # [{attribute, values:[{value,versions}]}]
    query: str = ""

    # ── render (raw evidence first, as the reviewer wants) ───────────────────
    def to_text(self, *, show_evidence: bool = True) -> str:
        head = {
            "ANSWERED": "✓",
            "PARTIAL": "◐ partial",
            "CONFLICT": "⚠ version conflict",
            "NOT_FOUND": "✗ not found",
            "INSUFFICIENT_EVIDENCE": "? insufficient evidence",
        }.get(self.status, self.status)
        lines = [f"[{head} · confidence {self.confidence}]"]
        if self.answer:
            lines.append(self.answer)
        if self.conflicts:
            lines.append("\n⚠ Conflicting values across versions (NOT guessed):")
            for c in self.conflicts:
                vs = "; ".join(f"{v['value']} (v{','.join(v.get('versions', []) or ['?'])})"
                               for v in c["values"][:4])
                lines.append(f"   {c['attribute']}: {vs}")
        if self.missing:
            lines.append("\nℹ Asked but NOT in the data: " + ", ".join(self.missing[:6]))
        if show_evidence and self.evidence:
            lines.append("\nEvidence (sources):")
            for e in self.evidence[:6]:
                tag = e.source + (f" v{e.version}" if e.version else "")
                if e.location and e.location != e.source:
                    tag += f" @{e.location}"
                # flag inferred evidence so the user never mistakes it for verbatim
                mark = " (inferred)" if e.kind == "inferred" else ""
                snippet = e.text[:200].replace("\n", " ")
                lines.append(f"   [{tag}]{mark} {snippet}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def not_found(query: str, msg: str = "") -> "AnswerResult":
        return AnswerResult(
            status="NOT_FOUND", query=query,
            answer=msg or ("I don't have that in the ingested data — nothing matched "
                           "with enough confidence."),
            confidence="LOW")
