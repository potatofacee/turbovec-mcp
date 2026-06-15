"""MCP server exposing turbovec-backed semantic code search.

Tools:
  - tv_search(query, k=10, path=".", compact=False)  semantic search
  - tv_index(path=".")                               (re)build the index
  - tv_status(path=".")                              index health

Run:  turbovec-mcp           (stdio transport, for MCP clients)
Embeddings come from an external OpenAI-compatible endpoint (see config.py).
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import store
from .config import load_config

mcp = FastMCP("turbovec-mcp")
_CFG = load_config()


@mcp.tool()
def tv_search(query: str, k: int = 10, path: str = ".", compact: bool = False) -> dict:
    """Semantic code search over an indexed repo.

    Args:
        query: natural-language or code-ish description of what to find.
        k: number of results.
        path: repo root that was indexed (default: current directory).
        compact: if true, omit chunk text (return only locations + scores).
    """
    hits = store.search(Path(path), query, k, _CFG, compact=compact)
    return {"query": query, "count": len(hits), "results": hits}


@mcp.tool()
def tv_index(path: str = ".") -> dict:
    """(Re)build the turbovec index for a repo. Run once before searching, and
    again after large code changes.

    Args:
        path: repo root to index (default: current directory).
    """
    return store.build(Path(path), _CFG)


@mcp.tool()
def tv_status(path: str = ".") -> dict:
    """Report whether a repo is indexed and basic index stats.

    Args:
        path: repo root (default: current directory).
    """
    return store.status(Path(path))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
