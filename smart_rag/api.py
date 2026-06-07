#!/usr/bin/env python3
"""SmartRAG — the one object users touch. First and last point for their data.

    d = SmartRAG()
    d.ingest("myfile.xlsx")        # or a folder, or many files
    d.ask("UFS and RAM for SKU1001")     # grounded AI-style answer + sources
    d.search(entity="SKU1001", attribute="ufs")   # programmatic, no LLM

Ingestion distills any supported format into one compact, provenance-tracked
FactStore (+ prose chunks for narrative). `ask` routes per query: entity-attribute
→ instant grounded fact answer; conceptual → prose/semantic; mixed → both. It
NEVER invents values — if a fact isn't there, it says so.

`ask` can optionally pass an answer through an LLM for natural phrasing, but the
GROUNDING (the values + sources) always comes from the distilled facts, so the
correctness/anti-hallucination guarantee holds with or without an LLM.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from smart_rag.core.fact import Fact, FactStore
from smart_rag.core.router import classify_query, extract_entity, attr_hint
from smart_rag.adapters import adapter_for




class SmartRAG:
    def __init__(self, db_path: "str | None" = None) -> None:
        """If db_path is given, use a durable SQLite store (persists across runs,
        incremental ingest by file hash). The in-memory FactStore stays the hot
        layer for fast lookups. Without db_path it's purely in-memory (+ jsonl
        save/load fallback). jsonl is kept until SQLite is proven clean."""
        self.store = FactStore()
        self.sources: List[str] = []
        self.db = None
        if db_path:
            from smart_rag.core.db import DistillDB
            self.db = DistillDB(db_path)
            self.db.load_into(self.store)   # hydrate hot layer from disk
            self._load_persisted_prose_index()   # use stored vectors, no re-embed

    def _load_persisted_prose_index(self) -> None:
        """Build the prose index from PERSISTED embeddings (no re-embedding on
        start). Only used when a DB exists and has vectors."""
        if self.db is None:
            return
        try:
            matrix, chunks = self.db.load_vectors()
            if matrix is not None and chunks:
                from smart_rag.core.prose_index import ProseIndex
                self.store.prose_index = ProseIndex(chunks, matrix=matrix)
        except Exception:
            pass

    # ── ingest ───────────────────────────────────────────────────────────────
    def ingest(self, path: str, *, recursive: bool = True, verbose: bool = True) -> dict:
        """Ingest a file or a folder (any supported formats). Returns stats.

        Prints progress + a coverage summary (what was ingested vs skipped) so a
        user can see exactly what happened on a real folder. Set verbose=False to
        silence.
        """
        paths = self._collect(path, recursive)
        return self.ingest_paths(paths, verbose=verbose,
                                 prune_under=path if os.path.isdir(path) else None)

    def ingest_paths(self, paths, *, verbose: bool = True, prune_under=None) -> dict:
        """Ingest a PRE-COLLECTED list of files, doing FTS rebuild + embedding ONCE
        at the end (not per file). This is the fast path for a large folder — calling
        ingest() per-file rebuilds FTS O(n) times → O(n²) and takes hours. Collectors
        pass their noise-filtered file list here."""
        from smart_rag.core.fact import source_id_for
        supported = [p for p in paths if adapter_for(p)]
        used = empty = skipped_unchanged = 0
        skipped_unsupported = len(paths) - len(supported)
        errors: List[str] = []
        if verbose and len(supported) > 1:
            print(f"[distill] ingesting {len(supported)} supported file(s) "
                  f"({skipped_unsupported} unsupported skipped)…")
        seen_ids = set()
        for i, p in enumerate(supported, 1):
            ad = adapter_for(p)
            sid = source_id_for(p)
            seen_ids.add(sid)
            try:
                h = ""
                if self.db is not None:
                    from smart_rag.core.db import content_hash
                    h = content_hash(p)
                    if self.db.source_unchanged(sid, h):
                        skipped_unchanged += 1
                        continue
                # PARSE + VALIDATE fully BEFORE committing (atomic). Stamp source_id
                # + location onto every fact so identity is stable.
                facts = [f.__class__(**{**f.__dict__, "source_id": sid,
                                        "location": f.location or f.source})
                         for f in ad.extract(p)]
                prose = [{**ch, "source_id": sid} for ch in ad.prose_chunks(p)]

                if self.db is not None:
                    # atomic replace by source_id (old rows for THIS file removed)
                    self.db.replace_source(sid, os.path.abspath(p), h, ad.name,
                                           facts, prose)
                # hot layer: drop this file's old rows, add fresh
                self.store.remove_source_id(sid)
                self.store.add_many(facts)
                for ch in prose:
                    self.store.add_prose(ch.get("text", ""), ch.get("source", ""),
                                         ch.get("version", ""), ch.get("title", ""),
                                         source_id=sid)
                if facts or prose:
                    used += 1
                else:
                    empty += 1   # HONEST: produced nothing → 'empty', not 'ingested'
                self.sources.append(p)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{os.path.basename(p)}: {e}")
                if self.db is not None:
                    # record the error but PRESERVE the prior version (no delete)
                    try:
                        self.db.replace_source(sid, os.path.abspath(p), h, ad.name,
                                               [], [], error=str(e)[:200])
                    except Exception:
                        pass
            if verbose and len(supported) > 5 and (i % 25 == 0 or i == len(supported)):
                print(f"[distill]   {i}/{len(supported)} files…")

        if self.db is not None:
            # A FULL-FOLDER ingest also removes DELETED sources (prune_under set).
            if prune_under and os.path.isdir(prune_under):
                base = os.path.abspath(prune_under)
                for gone in (self.db.known_source_ids() - seen_ids):
                    sp = self.db.source_path(gone) or ""
                    if sp and sp.startswith(base) and not os.path.exists(sp):
                        self.db.remove_source(gone)
                        self.store.remove_source_id(gone)
            # FTS + embeddings run for BOTH single-file and folder ingests (a
            # single-file ingest previously persisted ZERO vectors → no semantic
            # search after restart).
            self.db.rebuild_fts()
            self._embed_new_prose(verbose)

        st = self.store.stats()
        st["files_ingested"] = used
        st["files_empty"] = empty
        st["files_skipped_unsupported"] = skipped_unsupported
        st["files_skipped_unchanged"] = skipped_unchanged
        st["errors"] = errors
        if verbose:
            print(f"[distill] done: {used} ingested, {empty} empty, "
                  f"{skipped_unchanged} unchanged → {st['entities']:,} entities, "
                  f"{st['distinct_facts']:,} facts"
                  + (f", {len(errors)} errored" if errors else ""))
        return st

    def ingest_chunks(self, chunks, *, source_key: str = "source",
                      text_key: str = "text", verbose: bool = False) -> dict:
        """Ingest ALREADY-EXTRACTED chunks (e.g. SharePoint's _document_store) as
        prose — no folder/re-download needed. Each chunk is a dict with text +
        source (+ optional title/version/metadata). Builds the hybrid index so
        search_chunks() / answer() work. Idempotent per source_id (chunk hash)."""
        from smart_rag.core.fact import source_id_for
        n = 0
        for ch in chunks:
            text = ch.get(text_key, "")
            if not text or not text.strip():
                continue
            src = ch.get(source_key, "") or "sharepoint"
            # stable id per chunk so re-ingest replaces, not duplicates
            sid = source_id_for(f"{src}::{hash(text) & 0xffffffff}")
            title = ch.get("title") or ch.get("program_code") or src
            self.store.add_prose(text, src, ch.get("version", ""), title, source_id=sid)
            if self.db is not None:
                self.db.replace_source(sid, src, "", "sharepoint", [],
                                       [{"text": text, "title": title,
                                         "source": src, "version": ch.get("version", "")}])
            n += 1
        if self.db is not None:
            self.db.rebuild_fts()
            self._embed_new_prose(verbose)
        self.store.prose_index = None   # rebuild on next query
        if verbose:
            print(f"[distill] ingested {n} chunks → {len(self.store.prose)} prose total")
        return {"chunks_ingested": n, "prose_total": len(self.store.prose)}

    def search_chunks(self, query: str, top_k: int = 8):
        """Hybrid retrieval returning SharePoint-style strings: '[Source: X]\\ntext'.
        Drop-in for the SharePoint chat's search_documents(). No LLM.

        This store is PURE PROSE (no facts), so an id in the query is just a search
        term — NOT an entity to check existence against (don't abstain on it). We
        force a PROSE plan and apply the relevance floor for honest abstention."""
        from smart_rag.core.plan import QueryPlan, _STOP
        from smart_rag.core.retrieve import hybrid_retrieve, REL_FLOOR
        import re as _re
        topic = [w for w in _re.findall(r'\w+', query.lower())
                 if len(w) > 2 and w not in _STOP]
        plan = QueryPlan("PROSE", topic_terms=topic, raw=query)
        cands = hybrid_retrieve(plan, self.store, top_k=top_k)
        # honest abstention: keep only candidates clearing the absolute floor
        kept = [c for c in cands if c.abs_relevance >= REL_FLOOR]
        return [f"[Source: {c.source}]\n{c.text}" for c in kept]

    def _embed_new_prose(self, verbose: bool = False) -> None:
        """Persist embeddings for prose rows that don't have one yet (incremental —
        only changed/new sources get embedded, never the whole corpus on restart)."""
        if self.db is None:
            return
        from smart_rag.core import embed
        if not embed.available():
            return
        todo = self.db.prose_needing_vectors()
        if not todo:
            return
        if verbose:
            print(f"[distill] embedding {len(todo):,} new prose chunks…")
        B = 256
        for j in range(0, len(todo), B):
            batch = todo[j:j + B]
            texts = [f"{(t or s)}\n{txt}"[:2000] for (rid, txt, t) in
                     [(r[0], r[1], r[2]) for r in batch] for s in [t or ""]][: len(batch)]
            # simpler: build per-row text
            texts = [f"{(r[2] or '')}\n{r[1]}"[:2000] for r in batch]
            vecs = embed.embed(texts)
            self.db.store_vectors([(batch[k][0], vecs[k]) for k in range(len(batch))])

    @staticmethod
    def _collect(path: str, recursive: bool) -> List[str]:
        if os.path.isfile(path):
            return [path]
        out = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    out.append(os.path.join(root, f))
                if not recursive:
                    break
        return out

    # ── ask (text convenience wrapper over the v2 answer() pipeline) ─────────
    def ask(self, query: str, *, llm=None, max_attrs: int = 12) -> str:
        """Text convenience over answer(): runs the v2 plan→hybrid→verify pipeline
        and renders the structured AnswerResult to text. Optional `llm` only
        phrases the grounded result (grounding stays the source of truth). For
        programmatic use prefer answer() (returns the structured result)."""
        res = self.answer(query, max_attrs=max_attrs)
        text = res.to_text()
        if llm and res.status not in ("NOT_FOUND",):
            return self._phrase(query, text, llm)
        return text

    def answer(self, query: str, *, max_attrs: int = 12):
        """Structured, trustworthy answer via the v2 pipeline: PLAN → HYBRID
        RETRIEVE+RERANK → VERIFY. One calibrated scorer decides, so a weak entity
        match can never outrank a strong prose hit (the architectural cure for the
        recurring hijack). No LLM. Returns AnswerResult {status, evidence,
        confidence, missing, conflicts}."""
        from smart_rag.core.answer import AnswerResult, Evidence
        from smart_rag.core.plan import plan_query
        from smart_rag.core.retrieve import hybrid_retrieve, REL_FLOOR, MED_BAR, HIGH_BAR

        # 1. PLAN — classify once. UNKNOWN_ENTITY (precise id not in corpus) → NOT_FOUND.
        ent_set = set(self.store.entities)
        plan = plan_query(query, entity_exists=lambda e: e in ent_set)
        if plan.intent == "UNKNOWN_ENTITY":
            return AnswerResult(status="NOT_FOUND", query=query, confidence="LOW",
                                answer=f"'{plan.entity}' is not in the ingested data.")

        # RELATION (structural/edge): "what calls X", "what is signal S in".
        if plan.intent == "RELATION":
            rr = self._relation_answer(query, plan)
            if rr is not None:
                return rr   # else fall through to prose (description of the thing)

        # 2. RETRIEVE + RERANK on one scale.
        cands = hybrid_retrieve(plan, self.store, top_k=max(8, max_attrs))
        # ABSTENTION uses the ABSOLUTE relevance of the best candidate (cosine /
        # exact), NOT the normalized rank score — a weak keyword-only top hit
        # (e.g. "price"→"Cost") has low abs_relevance and is correctly rejected.
        best_abs = max((c.abs_relevance for c in cands), default=0.0)
        if not cands or best_abs < REL_FLOOR:
            return AnswerResult.not_found(query)

        # 3. FACT query about a known entity → facts + conflicts/missing.
        if plan.entity and plan.intent in ("FACT", "MIXED"):
            fr = self._fact_answer(query, plan, cands)
            if fr is not None:
                return fr

        # 4. PROSE / REVERSE / topic → reranked evidence; confidence from abs_relevance.
        conf = "HIGH" if best_abs >= HIGH_BAR else \
               ("MEDIUM" if best_abs >= MED_BAR else "LOW")
        # keep only candidates that clear the floor (don't pad with noise)
        kept = [c for c in cands if c.abs_relevance >= REL_FLOOR] or cands[:1]
        ev = [Evidence(text=c.text, source=c.source, version=c.version, score=c.abs_relevance)
              for c in kept[:5]]
        body = "\n\n".join(f"[{c.source}] {c.text}" for c in kept[:4])
        status = "ANSWERED" if conf in ("HIGH", "MEDIUM") else "INSUFFICIENT_EVIDENCE"
        return AnswerResult(status=status, query=query, confidence=conf,
                            evidence=ev, answer=body)

    def _relation_answer(self, query, plan):
        """Answer entity→entity edge questions (calls/imports/in_message/on_bus...)
        using relation facts (from the codegraph adapter / AUTOSAR). 'what calls X'
        = reverse walk; 'what does X call' = forward. Returns None to fall through."""
        from smart_rag.core.answer import AnswerResult, Evidence
        from smart_rag.core import relation as _rel
        ents = set(self.store.entities)
        # EXCLUDE the relation verb words from target candidates — 'call/calls/
        # import/contain/depend' describe the EDGE, not the entity. (On real code a
        # function may literally be named 'call', which wrongly won out before.)
        _verbs = {"call", "calls", "called", "import", "imports", "imported",
                  "contain", "contains", "depend", "depends", "reference",
                  "references", "use", "uses", "in", "on"}
        cands = [t for t in plan.topic_terms if t.lower() not in _verbs]
        # prefer an EXACT entity match over the longest term, so 'ACLHashPassword'
        # (exact symbol) beats a longer non-entity word.
        target = next((t for t in sorted(cands, key=len, reverse=True)
                       if t in ents), None)
        if target is None:
            for t in sorted(cands, key=len, reverse=True):
                hit = next((e for e in ents if t.lower() == e.lower()
                            or t.lower() in e.lower()), None)
                if hit:
                    target = hit; break
        if target is None:
            return None
        rel_attr = _rel.relation_attr_for(query)
        reverse = bool(__import__("re").search(r'\b(what|which|who)\b.*\bcall', query, __import__("re").I)) \
            and "called by" not in query.lower()
        # "what calls X" → reverse (who points at X); "what does X call" → forward
        forward = bool(__import__("re").search(r'\bdoes\b.*\b(call|import|depend|contain)', query, __import__("re").I))
        all_edges = _rel.walk_relations(self.store, target, rel_attr,
                                        reverse=not forward, limit=60)
        if not all_edges:
            return None
        # INTERNAL-FIRST: for forward "what does X call", lead with edges to your own
        # defined functions (internal), then external (libc/macros). Lets an agent see
        # the real code structure without library noise drowning it.
        internal = [e for e in all_edges if e.get("internal")]
        external = [e for e in all_edges if not e.get("internal")]
        edges = (internal + external)[:30]
        shown = internal if (forward and internal) else edges
        ext_note = ""
        if forward and internal and external:
            ext_note = (f"\n  (+ {len(external)} call(s) to external/library functions "
                        f"— ask 'including external' to see them)")
        ev = [Evidence(text=f"{e['from']} {e['rel']} {e['to']}", source=e["source"])
              for e in shown[:8]]
        # honor an explicit "including external/library/all" request (word-bounded,
        # so 'call' doesn't trigger on the 'all' substring).
        if __import__("re").search(r'\b(external|library|all|everything)\b',
                                   query, __import__("re").I):
            shown = edges; ext_note = ""
        body = (f"{target} — {('callers/sources' if not forward else 'targets')} "
                f"({len(shown)}):\n  " + "\n  ".join(
                    f"{e['from']} —{e['rel']}→ {e['to']}   [{e['source']}]"
                    for e in shown[:12]) + ext_note)
        return AnswerResult(status="ANSWERED", query=query, confidence="HIGH",
                            evidence=ev, answer=body)

    def _fact_answer(self, query, plan, cands):
        """Build a FACT AnswerResult for a known entity: resolved attributes with
        conflict + missing detection, each value cited. Returns None to fall through
        to the generic evidence path (e.g. when the concept side dominated)."""
        from smart_rag.core.answer import AnswerResult, Evidence
        from smart_rag.core import intent as _intent
        entity = plan.entity
        self.store.ensure_attr_index()
        res = _intent.resolve_attribute(query, self.store.attr_index,
                                        self.store.distinct_attributes())
        matched = res.get("matched") or []
        if not matched:
            return None
        ev, conflicts, present, lines = [], [], [], []
        for attr in matched:
            rows = self.store.lookup(entity, attr).get(attr)
            if not rows:
                continue
            present.append(attr); r0 = rows[0]
            lines.append(f"{attr}: {r0['value']}")
            ev.append(Evidence(text=f"{attr} = {r0['value']}",
                               source=(r0['sources'] or ['?'])[0],
                               version=(r0['versions'] or [''])[0]))
            if len(rows) > 1:
                conflicts.append({"attribute": attr,
                                  "values": [{"value": r["value"], "versions": r["versions"]}
                                             for r in rows]})
        missing = [a for a in matched if a not in present]
        if not lines:
            return None
        status = "CONFLICT" if conflicts else ("PARTIAL" if missing else "ANSWERED")
        conf = "HIGH" if res.get("confident") and not conflicts else \
               ("LOW" if missing and not present else "MEDIUM")
        # MIXED: also attach the top prose evidence so "specs AND how it works" works.
        if plan.intent == "MIXED":
            for c in cands:
                if c.kind == "prose" and c.score >= 0.3:
                    ev.append(Evidence(text=c.text, source=c.source, score=c.score))
                    break
        return AnswerResult(status=status, query=query,
                            answer=f"{entity}:\n  " + "\n  ".join(lines),
                            evidence=ev, confidence=conf,
                            missing=missing, conflicts=conflicts)

    def ask_compare(self, query: str, *, llm=None, max_attrs: int = 12) -> dict:
        """Return BOTH the raw grounded answer and (if llm given) an AI-phrased one,
        so the user can compare. {raw, ai}. Raw is always the source of truth."""
        raw = self.ask(query, llm=None, max_attrs=max_attrs)
        ai = None
        if llm and not raw.startswith("That isn't"):
            ai = self._phrase(query, raw, llm)
        return {"raw": raw, "ai": ai}

    def search(self, **kwargs):
        """Programmatic / keyword retrieval over facts (NO LLM, NO ranking) — the
        'last point of retrieval' surface for automation. Delegates to the store
        (entity=, attribute=, value_contains=, limit=)."""
        return self.store.search(**kwargs)

    def _phrase(self, query: str, grounding: str, llm) -> str:
        """Pass grounded facts through an LLM for natural phrasing. The facts are the
        ONLY source of truth — the prompt forbids adding anything not in them."""
        import asyncio
        prompt = (
            "Answer the user's question using ONLY the grounded facts below. Quote "
            "values exactly; cite the source. If a needed value isn't present, say so "
            "— do NOT invent or infer.\n\n"
            f"GROUNDED FACTS:\n{grounding}\n\nQUESTION: {query}")
        msgs = [{"role": "system", "content": "You are a precise, grounded data assistant."},
                {"role": "user", "content": prompt}]
        try:
            if asyncio.iscoroutinefunction(llm):
                ans = asyncio.new_event_loop().run_until_complete(llm(msgs))
            else:
                # Callables in this project take the MESSAGES list (not the raw
                # prompt string) — that's the contract used by the backends.
                ans = llm(msgs)
            ans = (ans or "").strip()
            # Only accept a REAL summary; if the model returned nothing/echoed the
            # grounding, fall back (so we never pretend AI ran when it didn't).
            if ans and ans[:80] != grounding[:80]:
                return ans
            return grounding
        except Exception as e:  # noqa: BLE001
            return grounding

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        open(path, "w", encoding="utf-8").write(self.store.to_jsonl())

    def load(self, path: str) -> None:
        self.store = FactStore.from_jsonl(open(path, encoding="utf-8").read())
