"""OpenAI-compatible embedding client. Bring your own /v1/embeddings endpoint."""
from __future__ import annotations

from typing import Iterable, Iterator

import httpx
import numpy as np

from .config import Config


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _headers(cfg: Config) -> dict[str, str]:
    # Only send Authorization when a key is set, so a blank key is never leaked.
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def _request(client: httpx.Client, cfg: Config, inputs: list[str]) -> list[list[float]]:
    resp = client.post(
        f"{cfg.endpoint}/embeddings",
        headers=_headers(cfg),
        json={"model": cfg.model, "input": inputs},
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(inputs):
        raise ValueError(
            f"embedding endpoint returned {len(data) if isinstance(data, list) else 'no'} "
            f"items for {len(inputs)} inputs"
        )
    # Preserve request order: OpenAI returns a per-item "index". Require it when
    # present for any item, else fall back to response order.
    if all("index" in d for d in data):
        data = sorted(data, key=lambda d: d["index"])
    try:
        return [d["embedding"] for d in data]
    except (KeyError, TypeError) as e:
        raise ValueError("embedding response item missing 'embedding' field") from e


def embed(texts: Iterable[str], cfg: Config, *, is_query: bool = False) -> np.ndarray:
    """Embed texts -> float32 array of shape (n, dim). Empty input -> (0, 0)."""
    prefix = cfg.query_prefix if is_query else cfg.doc_prefix
    inputs = [prefix + t for t in texts]
    if not inputs:
        return np.empty((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    with httpx.Client(timeout=cfg.timeout) as client:
        for batch in _batches(inputs, cfg.batch_size):
            vectors.extend(_request(client, cfg, batch))
    return np.asarray(vectors, dtype=np.float32)


def probe_dim(cfg: Config) -> int:
    """One round-trip to discover the embedding dimension."""
    return int(embed(["dimension probe"], cfg).shape[1])
