#!/usr/bin/env python3
"""Filesystem collector — walk a folder / drive / whole PC and yield ingestible
files, skipping the noise an agent never wants indexed (VCS, deps, caches, build
artifacts, binaries). Used by the index manager to build a workspace index fast.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

# Directories that are noise for retrieval — never walk into them.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    "dist", "build", "target", ".next", ".cache", ".gradle", "bin", "obj",
    ".smartrag",   # never index our own index store
}

# Extensions worth indexing by default (text/structured/standard formats).
# Media/archives skipped. NOTE: .so/.elf/.o are NOT skipped — the binary adapter
# extracts ELF symbols + strings from them now (built-in, no external codegraph).
_SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".zip", ".gz", ".7z", ".rar",
    ".tar", ".exe", ".dll", ".dylib", ".a", ".lib",
    ".pyc", ".pyo", ".class", ".jar", ".woff", ".woff2", ".ttf", ".eot",
}

_MAX_FILE_MB_DEFAULT = 25   # skip giant files unless raised


def collect_fs(root: str, *, max_file_mb: float = _MAX_FILE_MB_DEFAULT,
               skip_dirs: Optional[set] = None,
               skip_ext: Optional[set] = None,
               follow_symlinks: bool = False) -> Iterable[str]:
    """Yield absolute paths of ingestible files under `root`, skipping noise.

    Works for a folder, a whole drive (root='D:/'), or a home dir (whole-PC-ish).
    Streaming + os.scandir for speed on large trees.
    """
    skip_dirs = (skip_dirs or _SKIP_DIRS)
    skip_ext = (skip_ext or _SKIP_EXT)
    cap = max_file_mb * 1024 * 1024
    stack = [os.path.abspath(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=follow_symlinks):
                            if e.name.lower() in skip_dirs or e.name.startswith("."):
                                # skip hidden + noise dirs (but allow the root itself)
                                if e.name.lower() in skip_dirs:
                                    continue
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=follow_symlinks):
                            ext = os.path.splitext(e.name)[1].lower()
                            if ext in skip_ext:
                                continue
                            try:
                                if e.stat().st_size > cap:
                                    continue
                            except OSError:
                                continue
                            yield e.path
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue   # unreadable dir → skip, keep going (whole-PC walks hit these)
