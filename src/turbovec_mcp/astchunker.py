"""AST-aware chunking via tree-sitter.

Each function/class/method that fits the embedder becomes one coherent chunk on
clean boundaries. A god-class too large to embed whole is recursed into so its
methods become individual chunks. Any lines not covered by a definition (imports,
top-level code, or a giant leaf function that didn't fit) are line-windowed, so
coverage is total.

Returns None for unsupported languages / parse failures, so the caller can fall
back to plain line-window chunking.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .config import Config

# File extension -> tree-sitter-language-pack language name.
EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin", ".scala": "scala", ".swift": "swift",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".rb": "ruby", ".php": "php", ".lua": "lua", ".pl": "perl", ".pm": "perl",
    ".r": "r", ".jl": "julia", ".sh": "bash", ".bash": "bash",
    ".f": "fortran", ".f90": "fortran", ".f95": "fortran", ".for": "fortran",
}

# A node is a "definition" chunk boundary if its kind contains one of these.
_DEF_SUBSTR = (
    "function", "method", "class", "struct", "impl", "interface", "trait",
    "enum", "namespace", "subroutine", "program", "module", "constructor",
    "procedure",
)


def _is_def(kind: str) -> bool:
    return any(s in kind for s in _DEF_SUBSTR)


@lru_cache(maxsize=64)
def _parser(lang: str):
    from tree_sitter_language_pack import get_parser
    return get_parser(lang)


def _char_len(lines: list[str], s: int, e: int) -> int:
    return sum(len(lines[i]) + 1 for i in range(s, e + 1))


def _collect(node, lines: list[str], cfg: Config, out: list[tuple[int, int]]) -> None:
    """Append (start_row, end_row) 0-based ranges for definitions that fit."""
    if _is_def(node.kind()):
        s = node.start_position().row
        e = node.end_position().row
        if _char_len(lines, s, e) <= cfg.max_embed_chars:
            out.append((s, e))
            return  # whole coherent definition; do not descend further
        # too big: fall through and recurse so nested defs become chunks;
        # any leftover lines are covered by the gap filler in iter_ast_chunks.
    for i in range(node.named_child_count()):
        _collect(node.named_child(i), lines, cfg, out)


def def_ranges(text: str, lang: str, cfg: Config) -> Optional[list[tuple[int, int]]]:
    """0-based, sorted, non-overlapping (start_row, end_row) for definitions.

    None if the language is unsupported or parsing fails.
    """
    try:
        parser = _parser(lang)
        tree = parser.parse(text)
    except Exception:
        return None
    lines = text.splitlines()
    out: list[tuple[int, int]] = []
    root = tree.root_node()
    for i in range(root.named_child_count()):
        _collect(root.named_child(i), lines, cfg, out)
    out.sort()
    return out


def lang_for_suffix(suffix: str) -> Optional[str]:
    return EXT_LANG.get(suffix.lower())
