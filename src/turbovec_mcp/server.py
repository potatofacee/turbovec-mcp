"""MCP server exposing turbovec-backed semantic code search.

Tools:
  - tv_search(query, k=10, path=".")                 semantic search
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
def tv_search(query: str, k: int = 10, path: str = ".") -> dict:
    """Fuzzy SEMANTIC code discovery - find code when you can't name it.

    Use this for conceptual/"related to" questions where you do NOT know the
    symbol or file: "code related to the crossplot viewer", "where is retry
    backoff handled", "how is auth done". It matches by MEANING (embeddings),
    so it finds the right code even when the wording differs from the source.

    Pick the right tool:
      - You can NAME the symbol and want its definition, callers, or impact
        radius -> use codegraph (exact symbol graph), not this.
      - You need an exact literal string -> use grep, not this.
      - You're hunting a feature/concept by description and don't know where it
        lives -> THIS. It's the fuzzy-discovery tool the others can't replace.

    Returns TERSE hits - `score · path:start-end · signature line`, no source
    bodies - cheap to scan many (raise k for a broad survey). Step 1 of two:
    pick the locations that look right, then read them with tv_fetch. Do NOT
    open these files with Read; use tv_fetch.

    Args:
        query: the concept/behavior to find, described in natural language.
        k: number of hits to return (default 10; raise for a wider survey).
        path: repo root that was indexed (default: current directory).
    """
    hits = store.search(Path(path), query, k, _CFG)
    return {"query": query, "count": len(hits), "results": hits}


@mcp.tool()
def tv_fetch(locations: list[str], path: str = ".") -> dict:
    """Read the full source of chunks tv_search found. Step 2 of fuzzy discovery.

    Only used to follow up a tv_search: pass the `path:start-end` strings from
    its results for the hits you want to read. Returns verbatim source for each -
    treat it as ALREADY READ; do not re-open those files with the Read tool.
    Fetch only the few locations you actually need, not every hit.

    Args:
        locations: list of "path:start-end" strings (from tv_search results).
        path: repo root (default: current directory).
    """
    return {"results": store.fetch(Path(path), locations)}


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
        turbovec-mcp init [path]          # initialize a repo
        turbovec-mcp index <path>         # build/rebuild the index
        turbovec-mcp status <path>
        turbovec-mcp search <path> <query> [-k N]
        turbovec-mcp fetch <path> <location...>
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
    pn.add_argument("--no-register", action="store_true",
                    help="seed config only; skip writing .mcp.json")
    pi = sub.add_parser("index"); pi.add_argument("path")
    ps = sub.add_parser("status"); ps.add_argument("path")
    pq = sub.add_parser("search")
    pq.add_argument("path"); pq.add_argument("query")
    pq.add_argument("-k", type=int, default=10)
    pf = sub.add_parser("fetch")
    pf.add_argument("path"); pf.add_argument("locations", nargs="+")
    args = p.parse_args()

    if args.cmd is None:
        mcp.run()
        return

    if args.cmd == "init":
        from . import chunker, installer
        root = Path(args.path)

        cp = chunker.seed_config(root, _CFG, force=args.force)
        if cp is None:
            print(f"config exists: {chunker.config_path(root)} (use -f to regenerate)")
        else:
            sel = chunker.read_selection(root) or {"exclude": []}
            print(f"wrote {cp}  ({len(sel['exclude'])} exclude patterns seeded "
                  f"from default skips + nested .gitignores)")

        if not args.no_register:
            claude = installer.register_claude(root, _CFG, force=args.force)
            print(f".mcp.json (Claude Code): {claude}")
            oc = installer.register_opencode(root, _CFG, force=args.force)
            print(f"opencode.jsonc: {oc}")
            if oc == "manual":
                print("  existing opencode.jsonc has comments - paste this under "
                      "its \"mcp\" key:\n" + installer.opencode_snippet(_CFG))

        print(f"\nnext: turbovec-mcp index {args.path}")
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
        for h in store.search(Path(args.path), args.query, args.k, _CFG):
            loc = f"{h['path']}:{h['start_line']}-{h['end_line']}"
            print(f"{h['score']:.3f}  {loc}  {h['signature']}")
    elif args.cmd == "fetch":
        for r in store.fetch(Path(args.path), args.locations):
            if "error" in r:
                print(f"{r['location']}: {r['error']}")
                continue
            print(f"=== {r['path']}:{r['start_line']}-{r['end_line']} ===")
            print(r["text"])


if __name__ == "__main__":
    main()
