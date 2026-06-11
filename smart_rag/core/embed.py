#!/usr/bin/env python3
"""Self-contained embeddings for Smart RAG — GPU if present, else CPU, offline.

Kept inside the package so `distill/` is fully portable: copy the folder and intent
resolution works. Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) when
available; if it (or torch) is missing, callers fall back to keyword matching, so
the base tool still runs everywhere.
"""
from __future__ import annotations

import os
from typing import List, Optional

_MODEL = None
_AVAILABLE: Optional[bool] = None


def available() -> bool:
    """True if embeddings can ACTUALLY be used — i.e. the library imports AND the
    model loads (cached/offline). Verifying the load (not just the import) means a
    fresh PC without the cached model degrades GRACEFULLY to keyword-only retrieval
    instead of crashing mid-query. Cached after the first check."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401
            _model()                       # force the actual load; raises if absent
            _AVAILABLE = _MODEL is not None
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _pick_device() -> str:
    forced = os.environ.get("DISTILL_EMBED_DEVICE", "").lower().strip()
    if forced in ("cuda", "cpu"):
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _model_is_cached() -> bool:
    """True if all-MiniLM-L6-v2 is already in the HF cache (so we can run OFFLINE).
    If not cached, we must allow ONE online download (don't force offline)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # a core file of the model; present → cached
        hit = try_to_load_from_cache(
            "sentence-transformers/all-MiniLM-L6-v2", "config.json")
        return isinstance(hit, str) and os.path.exists(hit)
    except Exception:
        # can't tell → assume not cached so we don't wrongly force offline
        return False


def _model():
    global _MODEL
    if _MODEL is None:
        # Go OFFLINE only if the model is already cached. On a fresh PC the model
        # isn't cached → allow a ONE-TIME download (forcing offline made the work
        # PC permanently keyword-only even with internet + the library installed).
        if _model_is_cached():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            import logging
            for n in ("transformers", "sentence_transformers", "transformers.modeling_utils"):
                logging.getLogger(n).setLevel(logging.ERROR)
            import transformers
            transformers.logging.set_verbosity_error()
            try:
                transformers.logging.disable_progress_bar()
            except Exception:
                pass
        except Exception:
            pass
        # Last resort: the weight-materialization bar writes to stderr directly —
        # swallow stderr just during the one-time model construction.
        import contextlib
        import io
        try:
            from sentence_transformers import SentenceTransformer
            with contextlib.redirect_stderr(io.StringIO()):
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=_pick_device())
        except Exception:
            # Model not cached / offline / no torch → leave _MODEL None so the
            # system falls back to keyword retrieval (portable to a fresh PC).
            _MODEL = None
    return _MODEL


_WARNED_NO_MODEL = False


def embed(texts: List[str]):
    """Return a normalized numpy matrix [n, dim] (unit vectors → dot = cosine).

    When no embedding model is available (offline / no torch / model not
    cached) this returns ZERO vectors instead of crashing — the vector channel
    then contributes nothing and retrieval degrades to keyword/FTS, which is
    the documented fallback contract of _model().
    """
    import numpy as np
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    m = _model()
    if m is None:
        global _WARNED_NO_MODEL
        if not _WARNED_NO_MODEL:
            _WARNED_NO_MODEL = True
            print("  [embed] no embedding model available (offline/no torch) — "
                  "vector channel disabled, keyword retrieval only")
        return np.zeros((len(texts), 384), dtype="float32")
    batch = 256 if getattr(m, "device", None) and str(m.device).startswith("cuda") else 64
    return m.encode(texts, batch_size=batch, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True)


def embed_one(text: str):
    return embed([text])[0]
