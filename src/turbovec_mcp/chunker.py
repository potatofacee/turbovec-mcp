"""Walk a repo and yield line-window chunks. Language-agnostic v1.

A smarter AST-aware chunker is a future upgrade; fixed line windows with overlap
are a solid, simple baseline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import Config


@dataclass(frozen=True)
class Chunk:
    path: str          # relative to repo root
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    text: str


def _eligible(path: Path, cfg: Config) -> bool:
    if path.suffix.lower() not in cfg.extensions_resolved():
        return False
    try:
        if path.stat().st_size > cfg.max_file_mb * 1024 * 1024:
            return False
    except OSError:
        return False
    return True


def _iter_files(root: Path, cfg: Config) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in place so os.walk does not descend into them.
        dirnames[:] = [d for d in dirnames if d not in cfg.skip_dirs_resolved()]
        for name in filenames:
            p = Path(dirpath) / name
            if _eligible(p, cfg):
                yield p


def _windows(lines: list[str], cfg: Config) -> Iterator[tuple[int, list[str]]]:
    step = max(1, cfg.window - cfg.overlap)
    for start in range(0, len(lines), step):
        window = lines[start : start + cfg.window]
        if not window:
            break
        yield start, window
        if start + cfg.window >= len(lines):
            break


def iter_chunks(root: Path, cfg: Config) -> Iterator[Chunk]:
    for path in _iter_files(root, cfg):
        try:
            # "replace" keeps line boundaries intact on malformed bytes;
            # "ignore" can silently merge lines and shift line numbers.
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        lines = text.splitlines()
        rel = str(path.relative_to(root))
        for start, window in _windows(lines, cfg):
            body = "\n".join(window).strip()
            if not body:
                continue
            yield Chunk(
                path=rel,
                start_line=start + 1,
                end_line=start + len(window),
                text="\n".join(window),
            )
