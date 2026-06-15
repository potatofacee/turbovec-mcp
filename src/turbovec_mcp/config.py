"""Configuration, sourced from environment with sensible defaults.

Embeddings are external and OpenAI-compatible: point TURBOVEC_EMBED_ENDPOINT at
any /v1 base (your llama.cpp server, etc). The vector dimension is auto-detected
from the first embedding response, so it is never configured by hand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Dependency / build / cache / tooling-output directories, curated from the
# github/gitignore templates (mirrors codegraph's DEFAULT_IGNORE_DIRS). Excluded
# even with no .gitignore so the index is your code, not third-party noise.
# First-party-prone names (lib, app, bin, src, deps, packages) are deliberately
# omitted so real source is never hidden. Honored .gitignore handles the rest.
DEFAULT_SKIP_DIRS = frozenset({
    # VCS / our own data
    ".git", ".hg", ".svn", ".turbovec", ".codegraph",
    # JS / TS dependency dirs
    "node_modules", "bower_components", "jspm_packages", "web_modules",
    ".yarn", ".pnpm-store",
    # JS / TS framework & bundler output
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".vite", ".parcel-cache",
    ".angular", ".docusaurus", "storybook-static", ".vinxi", ".nitro",
    "out-tsc", ".vercel", ".netlify", ".wrangler",
    # Build output (cross-ecosystem)
    "dist", "build", "out", ".output",
    # Test / coverage
    "coverage", ".nyc_output",
    # Python
    "__pycache__", "__pypackages__", ".venv", "venv", ".pixi", ".pdm-build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".hypothesis", ".ipynb_checkpoints", ".eggs",
    # Rust / JVM (Maven, Gradle, Scala)
    "target", ".gradle",
    # .NET
    "obj",
    # Vendored deps (Go, PHP/Composer, Ruby/Bundler)
    "vendor",
    # Swift / iOS
    ".build", "Pods", "Carthage", "DerivedData", ".swiftpm",
    # Dart / Flutter
    ".dart_tool", ".pub-cache",
    # Native (Android NDK, C/C++ deps)
    ".cxx", ".externalNativeBuild", "vcpkg_installed",
    # Scala tooling
    ".bloop", ".metals",
    # Lua / Luau
    "lua_modules", ".luarocks",
    # Generic cache / IDE
    ".cache", ".idea", ".vscode",
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


# Sentinel: env var unset, so we auto-pick a prefix based on the model.
_AUTO = "\0auto"


@dataclass(frozen=True)
class Config:
    # Embedding endpoint (OpenAI-compatible). No trailing slash.
    endpoint: str = os.environ.get("TURBOVEC_EMBED_ENDPOINT", "http://127.0.0.1:8081/v1")
    model: str = os.environ.get("TURBOVEC_EMBED_MODEL", "nomic-embed-text-v1.5.Q8_0.gguf")
    api_key: str = os.environ.get("TURBOVEC_EMBED_API_KEY", "sk-local")
    # Task prefixes. nomic models want them; default to nomic's when the model
    # looks like nomic and the env var is unset, so a bare setup needs no config.
    doc_prefix: str = os.environ.get("TURBOVEC_DOC_PREFIX", _AUTO)
    query_prefix: str = os.environ.get("TURBOVEC_QUERY_PREFIX", _AUTO)

    # Embedding request batching + timeout.
    batch_size: int = _int("TURBOVEC_BATCH_SIZE", 64)
    timeout: float = float(os.environ.get("TURBOVEC_TIMEOUT", "120"))
    # Hard cap on chars sent per input so a chunk can't exceed the embedder's
    # context window (nomic ~2048 tokens) and 500. A resilient per-item shrink
    # in embedder.py is the backstop.
    max_embed_chars: int = _int("TURBOVEC_MAX_EMBED_CHARS", 1800)

    # turbovec quantization: 2 or 4 bits per coordinate (4 = better recall).
    bit_width: int = _int("TURBOVEC_BIT_WIDTH", 4)

    # Chunking: sliding window of lines with overlap.
    window: int = _int("TURBOVEC_CHUNK_LINES", 60)
    overlap: int = _int("TURBOVEC_CHUNK_OVERLAP", 12)
    max_file_mb: float = float(os.environ.get("TURBOVEC_MAX_FILE_MB", "2"))

    def __post_init__(self):
        # Resolve auto prefixes: nomic wants "search_document:/search_query:";
        # anything else defaults to none. Frozen dataclass -> object.__setattr__.
        nomic = "nomic" in self.model.lower()
        if self.doc_prefix == _AUTO:
            object.__setattr__(self, "doc_prefix", "search_document: " if nomic else "")
        if self.query_prefix == _AUTO:
            object.__setattr__(self, "query_prefix", "search_query: " if nomic else "")

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
