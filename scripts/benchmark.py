#!/usr/bin/env python3
"""Smart RAG vs competitors — a fair, runnable benchmark on YOUR data.

Five real methods, all computed on the SAME corpus + SAME labeled questions, so
the comparison is apples-to-apples (no vendor cherry-picking). Each method is what
it actually is — we don't fake a framework, we implement its CORE retrieval:

  RAW DUMP    — send the whole corpus to the LLM (the "just paste it" baseline).
  FLAT KEYWORD— chunk + word-overlap retrieval (naive RAG).
  BM25        — the classic IR ranking (Okapi BM25, implemented here, ~15 lines).
  VECTOR RAG  — chunk + embed + cosine top-k (the core of LangChain/LlamaIndex
                vector stores, using the SAME embedder Smart RAG uses — a fair fight).
  SMART RAG   — fact distillation + hybrid plan/rerank + abstention + citations.

Measured: ANSWER TOKENS (what you pay per query), CORRECTNESS (% labeled answered
right), ABSTENTION (rejects the unanswerable?), CITATIONS (can it cite a source?).

  python scripts/benchmark.py <path> --labels labels.csv --reject "q1;;q2"
  labels.csv rows:  query,expected_substring
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def toks(s): return len(_ENC.encode(s))
except Exception:
    def toks(s): return max(1, len(s) // 4)


# ── corpus → chunks (shared by all methods) ──────────────────────────────────
def load_chunks(path):
    from smart_rag.adapters import adapter_for
    low = path.lower()
    chunks = []
    if low.endswith((".xlsx", ".xlsm", ".csv")):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            for ws in wb.worksheets:
                rows = list(ws.iter_rows(values_only=True))
                if not rows:
                    continue
                cols = [str(c).strip() if c else f"c{i}" for i, c in enumerate(rows[0])]
                for r in rows[1:]:
                    cells = [f"{cols[i]}={r[i]}" for i in range(min(len(cols), len(r)))
                             if r[i] is not None and str(r[i]).strip()]
                    if cells:
                        chunks.append(" | ".join(cells))
            return chunks
        except Exception:
            pass
    if low.endswith((".log", ".txt", ".dlt", ".out")):
        for ln in open(path, encoding="utf-8", errors="replace"):
            if ln.strip():
                chunks.append(ln.rstrip())
        return chunks
    ad = adapter_for(path)
    try:
        for c in ad.prose_chunks(path):
            chunks.append(c.get("text", ""))
    except Exception:
        pass
    return chunks


# ── competitor retrievers ─────────────────────────────────────────────────────
def r_flat(chunks, q, k=8):
    terms = [w for w in _tok(q) if len(w) > 2]
    scored = [(sum(1 for w in terms if w in c.lower()), c) for c in chunks]
    return [c for s, c in sorted(scored, key=lambda x: -x[0]) if s > 0][:k]


import re as _re
def _tok(s):
    # word-boundary tokenize (so 'id=SKU1042' → ['id','sku1042'], a fair BM25)
    return _re.findall(r'\w+', s.lower())


def _bm25_index(chunks):
    docs = [_tok(c) for c in chunks]
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(N, 1)
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    return docs, idf, avgdl


def r_bm25(chunks, q, idx, k=8, k1=1.5, b=0.75):
    docs, idf, avgdl = idx
    terms = _tok(q)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in terms:
            if t in tf:
                s += idf.get(t, 0) * (tf[t] * (k1 + 1)) / (
                    tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append((s, chunks[i]))
    return [c for s, c in sorted(scores, key=lambda x: -x[0]) if s > 0][:k]


def r_vector(chunks, q, mat, embed_one, k=8, floor=0.25):
    import numpy as np
    qv = embed_one(q)
    sims = mat @ qv
    order = np.argsort(-sims)[:k]
    return [chunks[i] for i in order if sims[i] >= floor]


def _tfidf_index(chunks):
    """Classic TF-IDF vector space (sklearn) — a different IR baseline than BM25."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer()
        mat = vec.fit_transform(chunks)
        return vec, mat
    except Exception:
        return None


def r_tfidf(chunks, q, idx, k=8, floor=0.05):
    if idx is None:
        return []
    vec, mat = idx
    import numpy as np
    qv = vec.transform([q])
    sims = (mat @ qv.T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    return [chunks[i] for i in order if sims[i] >= floor]


def r_hybrid(chunks, q, bm25_idx, vec_mat, embed_one, k=8):
    """Hybrid BM25 + vector fused with Reciprocal Rank Fusion — what modern
    production RAG actually does. The STRONGEST flat-RAG competitor."""
    import numpy as np
    # BM25 ranking
    docs, idf, avgdl = bm25_idx
    terms = _tok(q)
    bm = []
    for i, d in enumerate(docs):
        from collections import Counter as _C
        tf = _C(d); s = 0.0
        for t in terms:
            if t in tf:
                s += idf.get(t, 0) * (tf[t] * 2.5) / (tf[t] + 1.5 * (0.25 + 0.75 * len(d) / avgdl))
        bm.append((s, i))
    bm_rank = {i: r for r, (s, i) in enumerate(sorted(bm, key=lambda x: -x[0])) if s > 0}
    # vector ranking
    vec_rank = {}
    if vec_mat is not None:
        sims = vec_mat @ embed_one(q)
        for r, i in enumerate(np.argsort(-sims)):
            if sims[i] >= 0.2:
                vec_rank[int(i)] = r
    # RRF fuse
    fused = {}
    for i, r in bm_rank.items():
        fused[i] = fused.get(i, 0) + 1.0 / (60 + r)
    for i, r in vec_rank.items():
        fused[i] = fused.get(i, 0) + 1.0 / (60 + r)
    top = sorted(fused, key=lambda i: -fused[i])[:k]
    return [chunks[i] for i in top]


# ── benchmark ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--labels"); ap.add_argument("--reject", default="")
    args = ap.parse_args()

    from smart_rag import SmartRAG
    chunks = load_chunks(args.path)
    raw_blob = "\n".join(chunks)
    labels = []
    if args.labels and Path(args.labels).exists():
        labels = [(r[0], r[1]) for r in csv.reader(open(args.labels, encoding="utf-8"))
                  if len(r) >= 2]
    rejects = [q for q in args.reject.split(";;") if q.strip()]

    sr = SmartRAG(); sr.ingest(args.path, verbose=False)
    bm25 = _bm25_index(chunks)
    tfidf = _tfidf_index(chunks)

    # optional vector baseline (same embedder Smart RAG uses)
    vec_mat = embed_one = None
    try:
        from smart_rag.core import embed
        if embed.available() and chunks:
            vec_mat = embed.embed(chunks)
            embed_one = embed.embed_one
    except Exception:
        pass

    methods = ["RAW DUMP", "FLAT KEYWORD", "TF-IDF", "BM25", "VECTOR RAG",
               "HYBRID (BM25+vec RRF)", "SMART RAG"]
    res = {m: {"atoks": [], "correct": 0, "abstain": 0, "cite": 0, "ans_n": 0}
           for m in methods}

    def record(m, ctx_tokens, correct, can_cite):
        res[m]["atoks"].append(ctx_tokens)
        res[m]["ans_n"] += 1
        res[m]["correct"] += correct
        res[m]["cite"] += can_cite

    for q, exp in labels:
        # RAW DUMP — whole corpus
        record("RAW DUMP", toks(raw_blob), exp.lower() in raw_blob.lower(), 0)
        # FLAT
        fc = r_flat(chunks, q); ft = "\n".join(fc)
        record("FLAT KEYWORD", toks(ft), exp.lower() in ft.lower(), 0)
        # TF-IDF
        tc = r_tfidf(chunks, q, tfidf); tt = "\n".join(tc)
        record("TF-IDF", toks(tt), exp.lower() in tt.lower(), 0)
        # BM25
        bc = r_bm25(chunks, q, bm25); bt = "\n".join(bc)
        record("BM25", toks(bt), exp.lower() in bt.lower(), 0)
        # VECTOR
        if vec_mat is not None:
            vc = r_vector(chunks, q, vec_mat, embed_one); vt = "\n".join(vc)
            record("VECTOR RAG", toks(vt), exp.lower() in vt.lower(), 0)
        # HYBRID (BM25 + vector, RRF)
        hc = r_hybrid(chunks, q, bm25, vec_mat, embed_one); ht = "\n".join(hc)
        record("HYBRID (BM25+vec RRF)", toks(ht), exp.lower() in ht.lower(), 0)
        # SMART RAG
        a = sr.answer(q); at = a.to_text()
        ok = exp.lower() in at.lower() and a.status in ("ANSWERED", "PARTIAL", "CONFLICT")
        record("SMART RAG", toks(at), ok, 1 if a.evidence else 0)

    # abstention (unanswerable)
    for m in methods:
        res[m]["rej_n"] = len(rejects)
    for q in rejects:
        # keyword/vector "abstain" only if they return nothing; raw never abstains
        res["RAW DUMP"]["abstain"] += 0
        res["FLAT KEYWORD"]["abstain"] += (0 if r_flat(chunks, q) else 1)
        res["TF-IDF"]["abstain"] += (0 if r_tfidf(chunks, q, tfidf) else 1)
        res["BM25"]["abstain"] += (0 if r_bm25(chunks, q, bm25) else 1)
        if vec_mat is not None:
            res["VECTOR RAG"]["abstain"] += (0 if r_vector(chunks, q, vec_mat, embed_one) else 1)
        res["HYBRID (BM25+vec RRF)"]["abstain"] += (0 if r_hybrid(chunks, q, bm25, vec_mat, embed_one) else 1)
        a = sr.answer(q)
        res["SMART RAG"]["abstain"] += (1 if a.status in ("NOT_FOUND", "INSUFFICIENT_EVIDENCE") else 0)

    # ── report ──
    def avg(xs): return round(sum(xs) / len(xs)) if xs else 0
    def pct(a, b): return f"{round(100*a/b)}%" if b else "—"
    print(f"\n=== Smart RAG vs competitors — {args.path} ===")
    print(f"corpus: {len(chunks):,} chunks · {toks(raw_blob):,} tokens · "
          f"{len(labels)} labeled queries · {len(rejects)} reject queries\n")
    hdr = f"{'method':<22}{'avg answer toks':>16}{'correct':>10}{'abstain':>10}{'cites?':>9}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        r = res[m]
        if m in ("VECTOR RAG", "HYBRID (BM25+vec RRF)") and vec_mat is None:
            print(f"{m:<22}{'(no embedder — skipped)':>43}"); continue
        cite = "yes" if r["cite"] else "no"
        print(f"{m:<22}{avg(r['atoks']):>16,}{pct(r['correct'],r['ans_n']):>10}"
              f"{pct(r['abstain'],r['rej_n']):>10}{cite:>9}")
    print("\nNotes: 'avg answer toks' = tokens sent to the LLM per query (lower=cheaper).")
    print("'cites?' = can the method attribute its answer to a source (groundable).")
    print("RAW DUMP / FLAT / BM25 / VECTOR return TEXT — they can't say 'I don't know'")
    print("with calibration, and none cite. Smart RAG answers, abstains, and cites.")


if __name__ == "__main__":
    main()
