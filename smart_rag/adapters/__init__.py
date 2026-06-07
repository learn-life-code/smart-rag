"""Format adapters. Register new ones here; the API dispatches by file suffix.

Order matters — first matching adapter wins. Code/docs/autosar are checked before
the generic logs/tabular so source/doc/config files route correctly. A .txt is
log-like → logs, else → docs (decided in adapter_for).
"""
import os
import re
from smart_rag.adapters.tabular import TabularAdapter
from smart_rag.adapters.logs import LogsAdapter
from smart_rag.adapters.docs import DocsAdapter
from smart_rag.adapters.code import CodeAdapter
from smart_rag.adapters.autosar import AutosarAdapter
from smart_rag.adapters.config import ConfigAdapter
from smart_rag.adapters.codegraph import CodegraphAdapter
from smart_rag.adapters.openapi import OpenAPIAdapter
from smart_rag.adapters.automotive_std import AutomotiveStdAdapter
from smart_rag.adapters.chip_std import ChipStdAdapter

_CODE = CodeAdapter()
_DOCS = DocsAdapter()
_AUTOSAR = AutosarAdapter()
_CONFIG = ConfigAdapter()
_CODEGRAPH = CodegraphAdapter()
_OPENAPI = OpenAPIAdapter()
_AUTO_STD = AutomotiveStdAdapter()
_CHIP = ChipStdAdapter()
_LOGS = LogsAdapter()
_TABULAR = TabularAdapter()

# Content-sniffed / specific adapters FIRST so .xml/.yaml/.json route to the right
# domain handler before the generic config/tabular catch them:
#   codegraph (.db nodes+edges) · openapi (yaml/json w/ openapi marker) ·
#   chip (.sp/.cir + ip-xact .xml) · automotive_std (.odx/.a2l) · autosar (.dbc/.arxml)
ADAPTERS = [_CODEGRAPH, _OPENAPI, _CHIP, _AUTO_STD, _AUTOSAR, _CODE, _DOCS,
            _CONFIG, _TABULAR, _LOGS]

_LOGLIKE = re.compile(
    r'\b\d{1,2}:\d{2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\[(ERROR|WARN|INFO|DEBUG)\]|'
    r'\b(error|fail|fatal|exception|qnx|dlt|logcat)\b', re.I)


def _txt_is_log(path: str) -> bool:
    """Sample the first lines: timestamps/severity → log; else doc/prose."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = "".join([next(fh, "") for _ in range(40)])
        hits = len(_LOGLIKE.findall(head))
        return hits >= 3
    except Exception:
        return False


def adapter_for(path: str):
    low = path.lower()
    if low.endswith(".txt"):
        return _LOGS if _txt_is_log(path) else _DOCS
    if low.endswith(".gz"):
        return _GZ
    for a in ADAPTERS:
        if a.can_handle(path):
            return a
    return None


class _GzAdapter:
    """Decompress a .gz on the fly and ingest its inner content (usually a log/
    text). The inner filename (sans .gz) decides which adapter parses it."""
    name = "gzip"
    suffixes = (".gz",)

    def can_handle(self, path):
        return path.lower().endswith(".gz")

    def _inner(self, path):
        import gzip
        import tempfile
        inner_name = os.path.basename(path)[:-3] or "data.txt"
        try:
            data = gzip.open(path, "rb").read()
        except Exception:
            return None, None
        tmp = os.path.join(tempfile.gettempdir(),
                           "distill_gz_" + re.sub(r'\W', '_', inner_name))
        try:
            open(tmp, "wb").write(data)
        except Exception:
            return None, None
        return tmp, inner_name

    def extract(self, path):
        tmp, _ = self._inner(path)
        if not tmp:
            return []
        inner = adapter_for(tmp.replace("distill_gz_", ""))  # route by inner ext
        inner = inner or _LOGS
        try:
            yield from inner.extract(tmp)
        finally:
            _safe_rm(tmp)

    def prose_chunks(self, path):
        tmp, _ = self._inner(path)
        if not tmp:
            return
        inner = adapter_for(tmp.replace("distill_gz_", "")) or _DOCS
        try:
            for c in inner.prose_chunks(tmp):
                yield c
        finally:
            _safe_rm(tmp)


def _safe_rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


_GZ = _GzAdapter()
