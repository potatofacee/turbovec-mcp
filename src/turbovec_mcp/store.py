"""Persisted index = a turbovec IdMapIndex + a JSON metadata sidecar.

Layout, per indexed repo:
    <repo>/.turbovec/index.tvim   - turbovec IdMapIndex (the compressed vectors)
    <repo>/.turbovec/meta.json    - {dim, model, bit_width, count, created, chunks}

`chunks` maps the integer id -> {path, start_line, end_line}. We do NOT store
chunk text: it is re-read from the file on demand for the handful of search
hits, which keeps meta.json small and the text always current. turbovec returns
ids on search; we look them up here to produce human-readable hits.

Each `build` writes a fresh index and a fresh meta, so ids and metadata never
drift out of sync (no incremental/stale-id problem).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from turbovec import IdMapIndex

from . import chunker
from .chunker import Chunk
from .config import Config
from .embedder import embed, probe_dim

# Cache loaded (index, meta) per repo, invalidated by the index file's mtime,
# so repeated searches do not reload a large index from disk every call.
_CACHE: dict[str, tuple[float, IdMapIndex, dict]] = {}


def index_dir(root: Path) -> Path:
    return root / ".turbovec"


def _paths(root: Path) -> tuple[Path, Path]:
    d = index_dir(root)
    return d / "index.tvim", d / "meta.json"


def build(root: Path, cfg: Config, *, progress=None) -> dict:
    """(Re)build the index for `root` from scratch. Returns a status dict.

    `progress`, if given, is called as progress(files_done, files_total) after
    each file so the caller can render a status line.
    """
    root = root.resolve()
    files = chunker.list_files(root, cfg)
    if not files:
        raise ValueError(f"no indexable files found under {root}")

    dim = probe_dim(cfg)
    if dim <= 0 or dim % 8 != 0:
        raise ValueError(
            f"embedding dimension {dim} is not a positive multiple of 8, which "
            f"turbovec requires - check TURBOVEC_EMBED_MODEL"
        )

    chunks: list[Chunk] = []
    vec_blocks: list[np.ndarray] = []
    total = len(files)
    for i, path in enumerate(files, 1):
        fchunks = chunker.chunks_for_file(path, root, cfg)
        if fchunks:
            vec_blocks.append(embed((c.text for c in fchunks), cfg))
            chunks.extend(fchunks)
        if progress:
            progress(i, total)
    if not chunks:
        raise ValueError(f"files found but no indexable content under {root}")

    vectors = np.vstack(vec_blocks)
    ids = np.arange(len(chunks), dtype=np.uint64)

    index = IdMapIndex(dim=dim, bit_width=cfg.bit_width)
    index.add_with_ids(vectors, ids)

    idx_path, meta_path = _paths(root)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    index.write(str(idx_path))

    # Store locations only - text is re-read on search (see module docstring).
    meta = {
        "dim": dim,
        "model": cfg.model,
        "bit_width": cfg.bit_width,
        "count": len(chunks),
        "created": time.time(),
        "root": str(root),
        "chunks": {
            str(i): {"path": c.path, "start_line": c.start_line, "end_line": c.end_line}
            for i, c in enumerate(chunks)
        },
    }
    meta_path.write_text(json.dumps(meta))
    _CACHE.pop(str(root), None)  # force reload on next search
    return {
        "indexed_chunks": len(chunks),
        "dim": dim,
        "model": cfg.model,
        "bit_width": cfg.bit_width,
        "index_path": str(idx_path),
    }


def _load_meta(root: Path) -> Optional[dict]:
    _, meta_path = _paths(root)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _load_cached(root: Path) -> tuple[IdMapIndex, dict]:
    """Load (index, meta), reusing the cache unless the index file changed."""
    idx_path, _ = _paths(root)
    if not idx_path.exists():
        raise ValueError(f"{root} is not indexed - run the `index` tool first")
    mtime = idx_path.stat().st_mtime
    key = str(root)
    cached = _CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    meta = _load_meta(root)
    if meta is None:
        raise ValueError(f"{root} is missing meta.json - re-run the `index` tool")
    index = IdMapIndex.load(str(idx_path))
    _CACHE[key] = (mtime, index, meta)
    return index, meta


def status(root: Path) -> dict:
    root = root.resolve()
    meta = _load_meta(root)
    if meta is None:
        return {"indexed": False, "root": str(root)}
    return {
        "indexed": True,
        "root": str(root),
        "count": meta["count"],
        "dim": meta["dim"],
        "model": meta["model"],
        "bit_width": meta["bit_width"],
        "created": meta["created"],
    }


def _read_slice(root: Path, rel: str, start: int, end: int) -> str:
    """Re-read lines [start, end] (1-based, inclusive) from the source file."""
    try:
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[start - 1 : end])


def _signature(root: Path, rel: str, start: int, end: int) -> str:
    """First non-blank line of a chunk - the triage hint for search results."""
    for line in _read_slice(root, rel, start, end).splitlines():
        if line.strip():
            return line.strip()
    return ""


def search(root: Path, query: str, k: int, cfg: Config) -> list[dict]:
    """Terse triage: nearest chunks as score + location + signature line.

    No source bodies - call fetch() with the locations you want to read.
    """
    root = root.resolve()
    index, meta = _load_cached(root)

    qvec = np.ascontiguousarray(embed([query], cfg, is_query=True), dtype=np.float32)
    # turbovec wants a (nq, dim) batch and returns (nq, k); we issue one query.
    k = max(1, min(k, meta["count"]))
    scores, ids = index.search(qvec, k=k)
    scores, ids = scores[0], ids[0]

    chunks = meta["chunks"]
    hits: list[dict] = []
    for score, cid in zip(scores, ids):
        c = chunks.get(str(int(cid)))
        if c is None:
            continue  # index/meta drift (shouldn't happen with full rebuilds)
        hits.append({
            "path": c["path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "score": float(score),
            "signature": _signature(root, c["path"], c["start_line"], c["end_line"]),
        })
    return hits


def fetch(root: Path, locations: list[str]) -> list[dict]:
    """Full source for explicit "path:start-end" locations from search results."""
    root = root.resolve()
    out: list[dict] = []
    for loc in locations:
        path, _, span = loc.rpartition(":")
        start_s, _, end_s = span.partition("-")
        try:
            start, end = int(start_s), int(end_s)
        except ValueError:
            out.append({"location": loc, "error": "expected path:start-end"})
            continue
        # Containment check: never read outside the repo root (absolute paths
        # and ../ escapes would otherwise reach any file on disk).
        if not (root / path).resolve().is_relative_to(root):
            out.append({"location": loc, "error": "outside repo"})
            continue
        out.append({
            "path": path,
            "start_line": start,
            "end_line": end,
            "text": _read_slice(root, path, start, end),
        })
    return out
