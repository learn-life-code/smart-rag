#!/usr/bin/env python3
"""Smart RAG CLI — point it at your data, ask, or search. No external RAG needed.

  python -m smart_rag.cli ingest <path> [--save store.jsonl]
  python -m smart_rag.cli ask    "<question>" [--store store.jsonl] [<path>]
  python -m smart_rag.cli search --entity SKU1001 --attr ufs [--store store.jsonl]
  python -m smart_rag.cli profile <path>
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# allow running from the repo without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smart_rag import SmartRAG          # noqa: E402
from smart_rag.core.profile import profile_items  # noqa: E402


def _load(args) -> SmartRAG:
    # --db = durable SQLite store (persists, incremental, instant reopen).
    d = SmartRAG(getattr(args, "db", None) or None)
    if getattr(args, "store", None) and Path(args.store).exists():
        d.load(args.store)   # jsonl fallback
    if getattr(args, "path", None):
        st = d.ingest(args.path)
        print(f"[ingested] {st['files_ingested']} files, {st['distinct_facts']:,} facts"
              + (f", {st.get('files_skipped_unchanged',0)} unchanged" if st.get('files_skipped_unchanged') else ""))
    return d


def main():
    ap = argparse.ArgumentParser(prog="distill")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("path"); p.add_argument("--save"); p.add_argument("--db")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("path", nargs="?")
    p.add_argument("--store"); p.add_argument("--db")
    p.add_argument("--no-evidence", action="store_true", help="hide the evidence/sources")
    p = sub.add_parser("search")
    p.add_argument("--entity"); p.add_argument("--attr"); p.add_argument("--value")
    p.add_argument("--store"); p.add_argument("--db"); p.add_argument("path", nargs="?")
    p.add_argument("--limit", type=int, default=30)
    p = sub.add_parser("profile"); p.add_argument("path"); p.add_argument("--db")

    # ── named persistent indexes (the agent workflow: build once, query fast) ──
    p = sub.add_parser("index", help="build/refresh a named index over a folder")
    p.add_argument("name"); p.add_argument("path")
    p.add_argument("--max-mb", type=float, default=25)
    p = sub.add_parser("query", help="query a named index (cited, abstains)")
    p.add_argument("name"); p.add_argument("question")
    p = sub.add_parser("indexes", help="list named indexes")
    p = sub.add_parser("drop", help="remove a named index"); p.add_argument("name")
    args = ap.parse_args()

    if args.cmd in ("index", "query", "indexes", "drop"):
        from smart_rag.collectors import IndexManager
        mgr = IndexManager()
        if args.cmd == "index":
            mgr.build(args.name, args.path, max_file_mb=args.max_mb)
        elif args.cmd == "query":
            print(mgr.answer(args.name, args.question).to_text())
        elif args.cmd == "indexes":
            cat = mgr.list()
            if not cat:
                print("No indexes. Build one: smart_rag index <name> <path>")
            for n, m in cat.items():
                print(f"  {n}: {m.get('entities')} entities, {m.get('facts')} facts "
                      f"← {m.get('source')}")
        elif args.cmd == "drop":
            print("removed" if mgr.remove(args.name) else "no such index")
        return

    if args.cmd == "ingest":
        d = SmartRAG(getattr(args, "db", None) or None)
        st = d.ingest(args.path)
        print(f"ingested: {st['files_ingested']} files → {st['distinct_facts']:,} facts, "
              f"{st['prose_chunks']:,} prose chunks"
              + (f", {st.get('files_skipped_unchanged',0)} unchanged" if st.get('files_skipped_unchanged') else ""))
        if args.db:
            print(f"durable store → {args.db} (reopen instantly next time)")
        if args.save:
            d.save(args.save); print(f"jsonl export → {args.save}")

    elif args.cmd == "ask":
        d = _load(args)
        # structured, trustworthy answer: status + confidence + evidence
        res = d.answer(args.question)
        print("\n" + res.to_text(show_evidence=not args.no_evidence))

    elif args.cmd == "search":
        d = _load(args)
        hits = d.search(entity=args.entity, attribute=args.attr,
                        value_contains=args.value, limit=args.limit)
        print(f"{len(hits)} fact(s):")
        for h in hits:
            print(f"  {h.entity} · {h.attribute}: {h.value}   [{h.source} v{h.version}]")

    elif args.cmd == "profile":
        d = SmartRAG(); d.ingest(args.path)
        # build a lightweight items view for the profiler
        items = [{"text": f"{f.entity} {f.attribute} {f.value}", "source": f.source}
                 for f in d.search(limit=20000)]
        items += [{"text": c["text"], "source": c["source"]} for c in d.store.prose]
        prof = profile_items(items)
        print("CORPUS PROFILE:", prof.summary())
        print("store stats:", d.store.stats())


if __name__ == "__main__":
    main()
