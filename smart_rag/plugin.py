#!/usr/bin/env python3
"""Distill plug-in surface — AI-FREE retrieval for other tools (RTC, SharePoint).

Distill is an independent tool. Other tools USE it as a plug-in: they call this to
get better-grounded, deduplicated, source-cited EVIDENCE, then feed it to THEIR OWN
AI. Distill does NO LLM work here — it only improves the grounding.

    r = DistillRetriever(folder="C:/.../_bugs/2905420_1")
    chunks = r.search("audio mute failure", top_k=8)   # cited evidence, no AI
    block  = r.evidence_block("audio mute failure")     # ready-to-inject prompt text

Or distill a whole folder into a compact fact/evidence summary for a forensic prompt:
    block = r.folder_evidence(max_chars=6000)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from smart_rag.api import SmartRAG


def _baseline_db_path(codegraph_path: str) -> "Path | None":
    """Where the baseline distill.db lives for a given build/codegraph path —
    factored out so the CLI and baseline_store_for agree on the location.

    Returns the FIRST existing candidate (so detection finds a baseline whether it
    was built from the build folder OR from the codegraph.db), else the preferred
    write location."""
    if not codegraph_path:
        return None
    p = Path(codegraph_path)
    if p.suffix.lower() == ".db":
        return p.with_name("baseline_distill.db")
    if p.is_dir():
        # check both known locations; prefer an existing one
        candidates = [p / ".distill" / "baseline_distill.db",
                      p / ".codegraph" / "baseline_distill.db"]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]   # default write location if none exist yet
    return None


def baseline_store_for(codegraph_path: str, *, build_if_missing: bool = True,
                       verbose: bool = False) -> "SmartRAG | None":
    """Graceful Distill-or-codegraph resolution for the RTC analyzer.

    Given the user's existing `codegraph_path` setting (a build root or a
    codegraph.db), return a persistent Distill BASELINE store that has ingested
    that software (including the codegraph.db → call-graph as RELATION facts).
    - If a baseline distill.db already exists next to it → reopen instantly.
    - Else build it once (ingest the folder/db) and persist.
    - Returns None if nothing usable → caller falls back to raw codegraph.

    So Distill is an ENHANCEMENT: present → richer (rerank/abstention/citations +
    absorbed call-graph); absent → caller uses codegraph directly. No hard dep.
    """
    if not codegraph_path:
        return None
    p = Path(codegraph_path)
    # Where to keep the baseline distill store (next to the codegraph).
    if p.suffix.lower() == ".db":          # pointed at a codegraph.db
        folder = p.parent.parent if p.parent.name == ".codegraph" else p.parent
        db_out = p.with_name("baseline_distill.db")
        ingest_target = str(p)             # at least absorb the codegraph.db
        also_folder = str(folder) if folder.is_dir() else None
    elif p.is_dir():                        # pointed at a build root
        db_out = p / ".distill" / "baseline_distill.db"
        ingest_target = str(p)
        also_folder = None
    else:
        return None
    try:
        db_out.parent.mkdir(parents=True, exist_ok=True)
        existed = db_out.exists()
        d = SmartRAG(str(db_out))
        if not existed and build_if_missing:
            if verbose:
                print(f"  [distill] building baseline store from {ingest_target}…")
            d.ingest(ingest_target, verbose=verbose)
            if also_folder and also_folder != ingest_target:
                d.ingest(also_folder, verbose=verbose)
        if d.store.entities:
            return d
        return None
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [distill] baseline store unavailable: {e}")
        return None


class DistillRetriever:
    """Thin, AI-free wrapper a consumer tool calls for distilled evidence."""

    def __init__(self, folder: Optional[str] = None, db_path: Optional[str] = None,
                 *, verbose: bool = False):
        self.d = SmartRAG(db_path)
        self.stats = {}
        if folder and os.path.isdir(folder):
            self.stats = self.d.ingest(folder, verbose=verbose)
        elif folder and os.path.isfile(folder):
            self.stats = self.d.ingest(folder, verbose=verbose)

    # ── retrieval (no AI) ─────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 8) -> List[dict]:
        """Return deduplicated, source-cited evidence for a query via the v2 HYBRID
        retriever (exact + vector + keyword, single-scale rerank). No LLM. Each
        item: {text, source, score, kind}. Works for prose-less stores (firmware
        baselines) because the keyword channel matches symbol/fact terms."""
        from smart_rag.core.plan import plan_query
        from smart_rag.core.retrieve import hybrid_retrieve
        ents = set(self.d.store.entities)
        plan = plan_query(query, entity_exists=lambda e: e in ents)
        cands = hybrid_retrieve(plan, self.d.store, top_k=top_k)
        out = [{"text": c.text[:600], "source": c.source,
                "score": c.abs_relevance, "kind": c.kind} for c in cands]
        # Fallback for symbol/string lookups with no rerank hit: direct entity
        # substring match (firmware strings like 'sbl_error_handler FAIL ...').
        if not out:
            ql = query.lower()
            terms = [w for w in ql.split() if len(w) > 2]
            for ent in self.d.store.entities:
                el = ent.lower()
                if ent == query or any(t in el for t in terms):
                    rows = self.d.store.lookup(ent)
                    a0 = next(iter(rows), "")
                    out.append({"text": f"{ent}" + (f" · {a0}: {rows[a0][0]['value']}"
                                                    if a0 else ""),
                                "source": (rows[a0][0]["sources"] or ["?"])[0] if a0 else "",
                                "score": 0.5, "kind": "fact"})
                    if len(out) >= top_k:
                        break
        return out[:top_k]

    def evidence_block(self, query: str, top_k: int = 8, max_chars: int = 6000) -> str:
        """A ready-to-inject, source-cited text block for a consumer's AI prompt."""
        hits = self.search(query, top_k)
        lines = ["=== DISTILLED EVIDENCE (deduplicated, source-cited) ==="]
        used = 0
        for h in hits:
            piece = f"[{h['source']}] {h['text']}"
            if used + len(piece) > max_chars:
                break
            lines.append(piece)
            used += len(piece)
        return "\n".join(lines) if len(lines) > 1 else ""

    def expectation_evidence(
        self,
        title: str,
        description: str = "",
        observed_hint: str = "",
        *,
        top_k_per_category: int = 4,
        max_chars: int = 9000,
        min_score: float = 0.12,
    ) -> str:
        """Retrieve typed baseline evidence for expected-versus-observed RCA.

        A single broad bug-text query tends to return whichever symbols happen to
        share the most words. These focused queries keep intended behavior,
        configuration, lifecycle flow, and ownership evidence distinguishable.
        """
        raw_focus = re.sub(
            r"\s+", " ", f"{title} {description} {observed_hint}"
        ).strip()
        tokens: list[str] = []
        token_lows: set[str] = set()
        stop = {
            "the", "and", "with", "from", "that", "this", "issue", "bug",
            "not", "working", "expected", "observed", "after", "when", "for",
            "into", "does", "did", "have", "has", "was", "were", "are",
            "missing", "setup", "learn", "data", "command", "function",
        }
        for token in re.findall(r"[A-Za-z0-9_./:-]{3,}", raw_focus):
            normalized = token.strip(".,;:()[]{}<>\"'")
            low = normalized.lower()
            if low in stop or low in token_lows:
                continue
            tokens.append(normalized)
            token_lows.add(low)
            if len(tokens) >= 18:
                break
        strong_anchor_words = {
            "fastboot", "fbl", "oem", "hud", "head-up", "headup", "b19b0",
            "calibration", "calibrationd", "service_vpe", "vpe", "wake",
            "resume", "wakeup", "bootloader", "abl", "xbl",
        }
        ticket_focus = f"{title} {description}".lower()
        anchor_candidates: list[tuple[int, str]] = []
        anchor_seen: set[str] = set()
        for index, token in enumerate(re.findall(r"[A-Za-z0-9_./:-]{3,}", raw_focus)):
            normalized = token.strip(".,;:()[]{}<>\"'")
            low = normalized.lower()
            if low in stop or low in anchor_seen:
                continue
            if (
                "_" in normalized
                or any(ch.isdigit() for ch in normalized)
                or normalized.isupper()
                or low in strong_anchor_words
            ):
                score = 1
                if low in ticket_focus:
                    score += 10
                if low in strong_anchor_words:
                    score += 8
                if normalized.isupper():
                    score += 4
                if "_" in normalized:
                    score += 4
                if any(ch.isdigit() for ch in normalized):
                    score += 3
                upper = normalized.upper()
                for marker in (
                    "HUD", "B19", "VPE", "FBL", "OEM", "FASTBOOT",
                    "SERVICE", "CALIB", "VEHICLE", "FRONT", "PRESENT",
                    "BOOTCONTROL",
                ):
                    if marker in upper:
                        score += 3
                if low.startswith(("var_log", "connected_camera")):
                    score -= 5
                anchor_candidates.append((score * 1000 - index, normalized))
                anchor_seen.add(low)
        anchors = [
            item for _score, item in sorted(anchor_candidates, reverse=True)[:24]
        ]
        lifecycle_anchors = {"wake", "resume", "wakeup"}
        domain_anchors = [
            anchor for anchor in anchors
            if anchor.lower() not in lifecycle_anchors
        ]

        def _anchor_matches(anchor: str, haystack: str) -> bool:
            if anchor.isupper() and len(anchor) <= 8 and anchor.isalnum():
                return bool(re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![a-z0-9])",
                    haystack,
                    re.I,
                ))
            return anchor.lower() in haystack.lower()

        focus = " ".join(tokens) or raw_focus[:300]
        categories = (
            (
                "INTENDED_BEHAVIOR",
                f"{focus} expected behavior requirement successful result acceptance",
            ),
            (
                "CONFIGURATION_AND_SUPPORT",
                f"{focus} configuration configured enabled supported command feature "
                "calibration mapping manifest",
            ),
            (
                "EXECUTION_FLOW_AND_PRECONDITIONS",
                f"{focus} call flow sequence initialization startup wake resume dependency "
                "producer consumer publish load read",
            ),
            (
                "OWNERSHIP_AND_BOUNDARY",
                f"{focus} owner ownership responsible vendor supplier Qualcomm Bosch GM "
                "module partition component",
            ),
        )
        category_hints = {
            "INTENDED_BEHAVIOR": (
                "expected", "requirement", "shall", "should", "feature",
                "successful", "acceptance",
            ),
            "CONFIGURATION_AND_SUPPORT": (
                "config", "enable", "supported", "unsupported", "command",
                "calibration", "mapping", "manifest", "oem", "variant",
            ),
            "EXECUTION_FLOW_AND_PRECONDITIONS": (
                "calls", "called", "depends", "function", "handler", "task",
                "service", "init", "load", "publish", "read", "write", "wake",
                "resume", "defined_in",
            ),
            "OWNERSHIP_AND_BOUNDARY": (
                "owned", "owner", "vendor", "supplier", "qualcomm", "bosch",
                "gm", "defined_in", ".elf", ".c", ".cpp", "partition", "img:",
            ),
        }

        lines = [
            "=== SOFTWARE BASELINE EXPECTATION EVIDENCE ===",
            "Reference evidence only: use it to verify intended behavior, configuration, "
            "execution flow, and ownership. A hit is not proof that the runtime event occurred.",
        ]
        used = sum(len(line) for line in lines)
        seen: set[tuple[str, ...]] = set()
        found = 0
        store = getattr(getattr(self, "d", None), "store", None)
        entity_by_lower = {
            entity.lower(): entity for entity in (store.entities if store else [])
        }

        def _anchor_hits(anchor: str) -> List[dict]:
            entity = entity_by_lower.get(anchor.lower())
            if entity is None:
                return self.search(anchor, top_k=top_k_per_category)
            out: list[dict] = []
            for attribute, rows in store.lookup(entity).items():
                for row in rows[:1]:
                    out.append({
                        "text": f"{entity} · {attribute}: {row.get('value', '')}",
                        "source": (row.get("sources") or ["?"])[0],
                        "score": 1.0,
                        "kind": row.get("kind") or "fact",
                    })
                    if len(out) >= top_k_per_category:
                        return out
            return out

        anchor_hit_batches = [
            (anchor, _anchor_hits(anchor)) for anchor in anchors[:12]
        ]
        for category, query in categories:
            hit_batches = [
                ("category", self.search(query, top_k=top_k_per_category))
            ] + anchor_hit_batches
            category_lines: list[str] = []
            for batch_name, hits in hit_batches:
                added_from_batch = 0
                for hit in hits:
                    text = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
                    source = str(hit.get("source", "") or "?")
                    if not text:
                        continue
                    score = float(hit.get("score", 0.0) or 0.0)
                    haystack = f"{text} {source}".lower()
                    if score < min_score:
                        continue
                    required_anchors = domain_anchors or anchors
                    if required_anchors and not any(
                        _anchor_matches(anchor, haystack)
                        for anchor in required_anchors
                    ):
                        continue
                    if not any(
                        hint in haystack for hint in category_hints[category]
                    ):
                        continue
                    key = (category, source.lower(), text[:240].lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    kind = str(hit.get("kind", "evidence"))
                    piece = (
                        f"  [{source}] ({kind}, score={score:.3f}) {text[:500]}"
                    )
                    if used + len(category) + len(piece) > max_chars:
                        break
                    category_lines.append(piece)
                    used += len(piece)
                    found += 1
                    added_from_batch += 1
                    if len(category_lines) >= top_k_per_category:
                        break
                    if batch_name != "category" and added_from_batch >= 1:
                        break
                if len(category_lines) >= top_k_per_category or used >= max_chars:
                    break
            if category_lines:
                lines.append(f"-- {category} --")
                lines.extend(category_lines)
                used += len(category)
            if used >= max_chars:
                break
        return "\n".join(lines) if found else ""

    # ── whole-folder distilled summary (for RTC bug-folder injection) ─────────
    def folder_evidence(self, max_chars: int = 6000, max_entities: int = 30) -> str:
        """Distill the WHOLE ingested folder into a compact fact/event summary —
        the high-signal, deduplicated view a forensic AI can ground on instead of
        raw 190k-line logs. Components → their notable events/facts, source-cited."""
        store = self.d.store
        if not store.entities:
            return ""
        # SIGNAL-FIRST ranking: prioritize entities carrying real diagnostic events
        # (error/crash/timeout/warning/nrc/hex from the logs adapter) over generic
        # metadata (filename/chars_fetched/total_lines). A high token-reduction is
        # only useful if it's the RIGHT content.
        _SIGNAL = {"error", "crash", "timeout", "warning", "uds_nrc", "hex_code", "event"}
        _NOISE_ATTR = {"filename", "chars_fetched", "truncated", "total_lines",
                       "summary", "kind"}

        def _signal_score(ent: str) -> tuple:
            facts = store.lookup(ent)
            sig = sum(1 for a in facts if a.lower() in _SIGNAL)
            real = sum(1 for a in facts if a.lower() not in _NOISE_ATTR)
            return (sig, real)   # sort by signal facts, then real facts

        ranked = sorted(store.entities, key=lambda e: _signal_score(e), reverse=True)
        ranked = [e for e in ranked if _signal_score(e)[1] > 0][:max_entities]
        lines = ["=== DISTILLED FOLDER EVIDENCE (components → key diagnostic "
                 "events/facts, deduplicated + source-cited; grounded evidence) ==="]
        used = len(lines[0])
        for ent in ranked:
            facts = store.lookup(ent)
            shown = 0
            for attr, rows in facts.items():
                if attr.lower() in _NOISE_ATTR:
                    continue
                val = rows[0]["value"]
                # skip values that are mostly binary/garbled (DLT artifacts)
                printable = sum(1 for c in val if 32 <= ord(c) < 127) / max(len(val), 1)
                if printable < 0.8:
                    continue
                piece = f"  {ent} · {attr}: {val[:120]}   [{(rows[0]['sources'] or ['?'])[0]}]"
                if used + len(piece) > max_chars:
                    return "\n".join(lines)
                lines.append(piece)
                used += len(piece)
                shown += 1
                if shown >= 4:
                    break
        return "\n".join(lines) if len(lines) > 1 else ""

    def summary(self) -> dict:
        s = self.d.store.stats()
        s.update({k: self.stats.get(k) for k in
                  ("files_ingested", "files_skipped_unsupported") if k in self.stats})
        return s
