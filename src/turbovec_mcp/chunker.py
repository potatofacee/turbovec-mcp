"""Walk a repo and yield line-window chunks. Language-agnostic v1.

A smarter AST-aware chunker is a future upgrade; fixed line windows with overlap
are a solid, simple baseline.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from . import astchunker
from .config import Config, DEFAULT_SKIP_DIRS


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


def _mk(rel: str, lines: list[str], s: int, e: int) -> Optional[Chunk]:
    """Build a Chunk from a 0-based inclusive row range, or None if blank."""
    body = "\n".join(lines[s : e + 1])
    if not body.strip():
        return None
    return Chunk(path=rel, start_line=s + 1, end_line=e + 1, text=body)


def _line_window_chunks(rel: str, lines: list[str], s: int, e: int, cfg: Config) -> Iterator[Chunk]:
    """Line-window a (0-based, inclusive) row span [s, e]."""
    sub = lines[s : e + 1]
    for off, window in _windows(sub, cfg):
        c = _mk(rel, lines, s + off, s + off + len(window) - 1)
        if c:
            yield c


def _ast_chunks(rel: str, lines: list[str], ranges: list[tuple[int, int]], cfg: Config) -> Iterator[Chunk]:
    """Emit definition chunks plus line-windowed chunks for uncovered gaps."""
    cursor = 0
    for s, e in ranges:
        if s > cursor:  # gap before this definition (imports, top-level code)
            yield from _line_window_chunks(rel, lines, cursor, s - 1, cfg)
        c = _mk(rel, lines, s, e)
        if c:
            yield c
        cursor = max(cursor, e + 1)
    if cursor <= len(lines) - 1:  # trailing gap
        yield from _line_window_chunks(rel, lines, cursor, len(lines) - 1, cfg)


# ----- selection: .turbovec/config.json is the source of truth -----
#
# Model: `init` walks every nested */.gitignore, rewrites their patterns
# workspace-relative, and bakes them (plus the built-in dep/build skips) into
# .turbovec/config.json as the SEED exclude list. After that, runtime reads ONLY
# the config - gitignore is never consulted live. Users edit the config freely.
# `include`, when non-empty, is a whitelist that can pull back anything excluded.

# Glob forms for the built-in skip dirs, matched anywhere in the tree.
_DEFAULT_SKIP_GLOBS = sorted(f"**/{d}/**" for d in DEFAULT_SKIP_DIRS)


def config_path(root: Path) -> Path:
    return root / ".turbovec" / "config.json"


def read_selection(root: Path) -> Optional[dict]:
    """{"include": [...], "exclude": [...]} from the config, or None if absent."""
    try:
        data = json.loads(config_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        "include": [g for g in (data.get("include") or []) if isinstance(g, str)],
        "exclude": [g for g in (data.get("exclude") or []) if isinstance(g, str)],
    }


def _match_any(rel_posix: str, globs) -> bool:
    return any(fnmatch.fnmatch(rel_posix, g) for g in globs)


def _prune_bases(exclude) -> list[str]:
    """Dir bases (exclude globs minus a trailing /** or /) for os.walk pruning.

    Pruning is best-effort speed only; the per-file exclude check is what's
    authoritative, so an imperfect prune never changes the result, just the cost.
    """
    bases = []
    for g in exclude:
        b = g[:-3] if g.endswith("/**") else (g[:-1] if g.endswith("/") else g)
        if b:
            bases.append(b)
    return bases


def list_files(root: Path, cfg: Config) -> list[Path]:
    """Indexable files under root.

    Config-driven when .turbovec/config.json exists (include/exclude globs are
    authoritative). Otherwise a safe default walk with the built-in skip dirs -
    run `init` to materialize an editable config seeded from your .gitignores.
    """
    sel = read_selection(root)
    if sel is None:  # no config yet: safe default (built-in skips + ext/size)
        return [p for p in _iter_files(root, cfg)
                if _within_size(p, cfg.max_file_mb)]

    include = tuple(sel["include"])
    exclude = tuple(sel["exclude"])
    bases = _prune_bases(exclude)
    exts = cfg.extensions_resolved()
    limit = cfg.max_file_mb * 1024 * 1024

    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        # Prune .git always, and any child dir matching an exclude base (speed).
        kept = []
        for d in dirnames:
            if d == ".git":
                continue
            child = (rel_dir / d).as_posix()
            if any(fnmatch.fnmatch(child, b) for b in bases):
                continue
            kept.append(d)
        dirnames[:] = kept

        for name in filenames:
            if Path(name).suffix.lower() not in exts:
                continue
            rel_s = (rel_dir / name).as_posix()
            if include and not _match_any(rel_s, include):
                continue
            if exclude and _match_any(rel_s, exclude):
                continue
            p = root / rel_s
            if not _within_size(p, cfg.max_file_mb):
                continue
            out.append(p)
    return out


def _within_size(p: Path, max_mb: float) -> bool:
    try:
        return p.is_file() and p.stat().st_size <= max_mb * 1024 * 1024
    except OSError:
        return False


# ----- init: seed .turbovec/config.json from nested .gitignores -----

def _iter_gitignores(root: Path, cfg: Config) -> Iterator[Path]:
    skip = cfg.skip_dirs_resolved()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and d != ".git"]
        if ".gitignore" in filenames:
            yield Path(dirpath) / ".gitignore"


def _rewrite_gitignore_pattern(pat: str, rel_dir: str) -> list[str]:
    """A nested .gitignore line -> workspace-relative exclude glob(s).

    Negations are dropped from the seed (a flat exclude list can't express them;
    re-add via `include`). Anchored (`/x`) patterns bind to the gitignore's dir;
    unanchored ones match at any depth beneath it.
    """
    pat = pat.strip()
    if not pat or pat.startswith("#") or pat.startswith("!"):
        return []
    anchored = pat.startswith("/")
    pat = pat.strip("/")
    if not pat:
        return []
    prefix = "" if rel_dir in ("", ".") else rel_dir + "/"
    if anchored:
        stems = [prefix + pat]
    else:  # match the name at any depth under rel_dir
        stems = [prefix + pat, prefix + "**/" + pat]
    out: list[str] = []
    for s in stems:
        out.append(s)          # the path itself
        out.append(s + "/**")  # and its subtree, if it's a dir
    return out


def seed_config(root: Path, cfg: Config, *, force: bool = False) -> Optional[Path]:
    """Write .turbovec/config.json seeded from default skips + nested gitignores.

    Returns the path, or None if a config already exists and force is False.
    """
    cp = config_path(root)
    if cp.exists() and not force:
        return None

    excludes: set[str] = set(_DEFAULT_SKIP_GLOBS)
    for gi in _iter_gitignores(root, cfg):
        rel_dir = gi.parent.relative_to(root).as_posix()
        try:
            lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            excludes.update(_rewrite_gitignore_pattern(line, rel_dir))

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps({"include": [], "exclude": sorted(excludes)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return cp


def chunks_for_file(path: Path, root: Path, cfg: Config) -> list[Chunk]:
    try:
        # "replace" keeps line boundaries intact on malformed bytes;
        # "ignore" can silently merge lines and shift line numbers.
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text.strip():
        return []
    lines = text.splitlines()
    rel = str(path.relative_to(root))

    lang = astchunker.lang_for_suffix(path.suffix)
    ranges = astchunker.def_ranges(text, lang, cfg) if lang else None
    if ranges:  # AST-aware: clean boundaries + gap fill
        return list(_ast_chunks(rel, lines, ranges, cfg))
    return list(_line_window_chunks(rel, lines, 0, len(lines) - 1, cfg))


def iter_chunks(root: Path, cfg: Config) -> Iterator[Chunk]:
    for path in _iter_files(root, cfg):
        yield from chunks_for_file(path, root, cfg)
