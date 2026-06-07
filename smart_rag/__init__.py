"""Smart RAG — a universal data-distillation layer: first and last point for your data.

Point it at files/logs/folders → a compact, deduplicated, provenance- and
version-tracked fact store → ask grounded questions or retrieve programmatically,
with far fewer tokens, smaller index, and correct (source-cited) answers.

    from smart_rag import SmartRAG
    d = SmartRAG(); d.ingest("data.xlsx"); print(d.ask("UFS for SKU1001"))
"""
from smart_rag.api import SmartRAG
from smart_rag.core.fact import Fact, FactStore

__all__ = ["SmartRAG", "Fact", "FactStore"]
__version__ = "0.1.0"
