#!/usr/bin/env python3
"""Download the embedding model once (needs internet). Run on a fresh PC to enable
HYBRID semantic retrieval. After this, Smart RAG runs fully offline.

  python scripts/smart_rag_fetch_model.py

If it fails, this prints WHICH kind of failure (version conflict / network / etc.)
and the exact fix — so you don't go down a rabbit hole.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Do NOT force offline here — we WANT to download.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)


def main():
    print("Downloading all-MiniLM-L6-v2 (one-time, ~90 MB)…")
    # 1. library import — catches the transformers/sentence-transformers conflict
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"❌ Library import failed: {msg}\n")
        if "PreTrainedModel" in msg or "transformers" in msg.lower():
            print("   CAUSE: incompatible transformers/sentence-transformers versions.")
            print("   FIX:  pip install -r smart_rag/requirements.txt   (it now pins a")
            print("         compatible, wheel-available combo for Python 3.13).")
            print("   Do NOT downgrade tokenizers to 0.19 — it has no 3.13 wheel and")
            print("   would try to compile Rust (needs Visual C++ link.exe).")
        else:
            print("   FIX:  pip install -r smart_rag/requirements.txt")
        sys.exit(1)

    # 2. the actual download
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        v = m.encode(["test"], normalize_embeddings=True)
        print(f"\n✅ Model ready (dim={v.shape[1]}). Smart RAG will now use HYBRID retrieval.")
        print("   Re-run:  python scripts/smart_rag_doctor.py   to confirm.")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"\n❌ Download failed: {msg}\n")
        low = msg.lower()
        if any(k in low for k in ("proxy", "ssl", "connection", "timed out",
                                  "max retries", "resolve", "403", "407", "certificate")):
            print("   CAUSE: network/corporate proxy is blocking huggingface.co.")
            print("   FIX (pick one):")
            print("     a) set a proxy:  $env:HTTPS_PROXY='http://your.proxy:port'  then re-run")
            print("     b) COPY the model from a PC where it works — copy the folder")
            print("        %USERPROFILE%\\.cache\\huggingface\\hub  to this PC (same path).")
            print("        Then Smart RAG runs offline, no download needed.")
        else:
            print("   CAUSE: unexpected. Smart RAG still works in KEYWORD-ONLY mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
