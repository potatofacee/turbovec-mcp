"""Configuration, sourced from environment with sensible defaults.

Embeddings are external and OpenAI-compatible: point TURBOVEC_EMBED_ENDPOINT at
any /v1 base (your llama.cpp server, etc). The vector dimension is auto-detected
from the first embedding response, so it is never configured by hand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Directories / files commonly worth skipping when walking a repo.
DEFAULT_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", "target", ".turbovec", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode", ".cache",
})

# Text/code extensions to index. Everything else is skipped.
DEFAULT_EXTENSIONS = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".cs", ".rb", ".php", ".swift",
    ".scala", ".sh", ".bash", ".zsh", ".fish", ".lua", ".pl", ".pm", ".r",
    ".jl", ".f", ".f90", ".f95", ".for", ".m", ".mm", ".sql", ".html", ".css",
    ".scss", ".vue", ".svelte", ".md", ".rst", ".txt", ".toml", ".yaml",
    ".yml", ".json", ".ini", ".cfg", ".dockerfile", ".tf", ".proto",
})


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Embedding endpoint (OpenAI-compatible). No trailing slash.
    endpoint: str = os.environ.get("TURBOVEC_EMBED_ENDPOINT", "http://127.0.0.1:8081/v1")
    model: str = os.environ.get("TURBOVEC_EMBED_MODEL", "nomic-embed-text-v1.5.Q8_0.gguf")
    api_key: str = os.environ.get("TURBOVEC_EMBED_API_KEY", "sk-local")
    # Some embedders (nomic) want task prefixes. Empty = none.
    doc_prefix: str = os.environ.get("TURBOVEC_DOC_PREFIX", "")
    query_prefix: str = os.environ.get("TURBOVEC_QUERY_PREFIX", "")

    # Embedding request batching + timeout.
    batch_size: int = _int("TURBOVEC_BATCH_SIZE", 64)
    timeout: float = float(os.environ.get("TURBOVEC_TIMEOUT", "120"))

    # turbovec quantization: 2 or 4 bits per coordinate (4 = better recall).
    bit_width: int = _int("TURBOVEC_BIT_WIDTH", 4)

    # Chunking: sliding window of lines with overlap.
    window: int = _int("TURBOVEC_CHUNK_LINES", 60)
    overlap: int = _int("TURBOVEC_CHUNK_OVERLAP", 12)
    max_file_mb: float = float(os.environ.get("TURBOVEC_MAX_FILE_MB", "2"))

    def extensions_resolved(self) -> frozenset[str]:
        extra = os.environ.get("TURBOVEC_EXTRA_EXTENSIONS", "")
        if not extra:
            return DEFAULT_EXTENSIONS
        more = {e if e.startswith(".") else f".{e}" for e in extra.split(",") if e.strip()}
        return DEFAULT_EXTENSIONS | more

    def skip_dirs_resolved(self) -> frozenset[str]:
        extra = os.environ.get("TURBOVEC_EXTRA_SKIP_DIRS", "")
        if not extra:
            return DEFAULT_SKIP_DIRS
        return DEFAULT_SKIP_DIRS | {d.strip() for d in extra.split(",") if d.strip()}


def load_config() -> Config:
    return Config()
