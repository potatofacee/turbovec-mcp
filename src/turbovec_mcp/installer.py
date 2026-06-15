"""Register the turbovec MCP server into a project's agent config.

Claude Code reads a project-local `.mcp.json`; we merge our entry in without
touching other servers. opencode reads `opencode.jsonc` (comment-bearing) - we
print a snippet to paste rather than risk mangling comments with a naive rewrite.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import Config


def self_command() -> str:
    """Absolute path to the running turbovec-mcp entry script.

    Reliable before the package is on PyPI (uvx/pipx can't resolve it yet).
    """
    exe = shutil.which("turbovec-mcp") or sys.argv[0]
    return str(Path(exe).resolve())


def mcp_env(cfg: Config) -> dict:
    env = {
        "TURBOVEC_EMBED_ENDPOINT": cfg.endpoint,
        "TURBOVEC_EMBED_MODEL": cfg.model,
    }
    if cfg.api_key:
        env["TURBOVEC_EMBED_API_KEY"] = cfg.api_key
    if cfg.doc_prefix:
        env["TURBOVEC_DOC_PREFIX"] = cfg.doc_prefix
    if cfg.query_prefix:
        env["TURBOVEC_QUERY_PREFIX"] = cfg.query_prefix
    return env


def register_claude(root: Path, cfg: Config, *, force: bool = False) -> str:
    """Merge a turbovec entry into <root>/.mcp.json. Returns added/exists.

    Preserves any other servers and keys already in the file.
    """
    f = root / ".mcp.json"
    data: dict = {}
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    servers = data.setdefault("mcpServers", {})
    if "turbovec" in servers and not force:
        return "exists"
    servers["turbovec"] = {
        "command": self_command(),
        "args": [],
        "env": mcp_env(cfg),
    }
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return "added"


def _opencode_entry(cfg: Config) -> dict:
    return {
        "type": "local",
        "command": [self_command()],
        "environment": mcp_env(cfg),
        "enabled": True,
    }


def opencode_snippet(cfg: Config) -> str:
    """opencode.jsonc block to paste under the top-level "mcp" key."""
    return json.dumps({"turbovec": _opencode_entry(cfg)}, indent=2)


def _strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments (outside strings) so JSONC parses as JSON."""
    out, i, n = [], 0, len(text)
    instr = esc = False
    while i < n:
        c = text[i]
        if instr:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            i += 1
            continue
        if c == '"':
            instr = True; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c); i += 1
    return "".join(out)


def register_opencode(root: Path, cfg: Config, *, force: bool = False) -> str:
    """Register turbovec in project-local <root>/opencode.jsonc.

    Returns: "created" (new file), "added", "exists", or "manual" (a comment-
    bearing file we won't risk rewriting - caller should print the snippet).
    """
    f = root / "opencode.jsonc"
    if not f.exists():
        f.write_text(
            json.dumps(
                {"$schema": "https://opencode.ai/config.json",
                 "mcp": {"turbovec": _opencode_entry(cfg)}},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return "created"

    raw = f.read_text(encoding="utf-8")
    # Only rewrite when the file has no comments (else we'd drop them).
    if _strip_jsonc(raw).strip() != raw.strip():
        return "manual"
    try:
        data = json.loads(raw)
    except ValueError:
        return "manual"
    mcp = data.setdefault("mcp", {})
    if "turbovec" in mcp and not force:
        return "exists"
    mcp["turbovec"] = _opencode_entry(cfg)
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return "added"
