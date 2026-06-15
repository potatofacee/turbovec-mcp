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
    """Entry point. No args -> run the MCP server (stdio). Subcommands let you
    drive it from the terminal for testing:

        turbovec-mcp                      # MCP server (for clients)
        turbovec-mcp index <path>         # build/rebuild the index
        turbovec-mcp status <path>
        turbovec-mcp search <path> <query> [-k N] [--compact]
    """
    import argparse
    import json
    import logging

    # Quiet httpx's per-request INFO lines; embedding does thousands of calls.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(prog="turbovec-mcp")
    sub = p.add_subparsers(dest="cmd")
    pn = sub.add_parser("init"); pn.add_argument("path", nargs="?", default=".")
    pn.add_argument("-f", "--force", action="store_true")
    pi = sub.add_parser("index"); pi.add_argument("path")
    ps = sub.add_parser("status"); ps.add_argument("path")
    pq = sub.add_parser("search")
    pq.add_argument("path"); pq.add_argument("query")
    pq.add_argument("-k", type=int, default=10)
    pq.add_argument("--compact", action="store_true")
    args = p.parse_args()

    if args.cmd is None:
        mcp.run()
        return

    if args.cmd == "init":
        from . import chunker
        cp = chunker.seed_config(Path(args.path), _CFG, force=args.force)
        if cp is None:
            print(f"config already exists: {chunker.config_path(Path(args.path))} "
                  f"(use --force to regenerate)")
        else:
            sel = chunker.read_selection(Path(args.path)) or {"exclude": []}
            print(f"wrote {cp}\n  {len(sel['exclude'])} exclude patterns seeded "
                  f"from default skips + nested .gitignores\n"
                  f"  edit it, then: turbovec-mcp index {args.path}")
        return

    if args.cmd == "index":
        import sys
        import time

        start = time.time()

        def fmt(secs: float) -> str:
            secs = int(secs)
            return f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"

        def progress(done: int, total: int) -> None:
            pct = done * 100 // total
            elapsed = time.time() - start
            eta = f"  {fmt(elapsed / done * (total - done))} left" if done else ""
            end = "\n" if done == total else ""
            print(f"\rscanned {done}/{total} files  {pct}%{eta}   ", end=end, file=sys.stderr, flush=True)

        print(json.dumps(store.build(Path(args.path), _CFG, progress=progress), indent=2))
    elif args.cmd == "status":
        print(json.dumps(store.status(Path(args.path)), indent=2))
    elif args.cmd == "search":
        hits = store.search(Path(args.path), args.query, args.k, _CFG, compact=args.compact)
        for h in hits:
            print(f"{h['score']:.3f}  {h['path']}:{h['start_line']}-{h['end_line']}")
            if not args.compact:
                print("    " + h["text"].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
