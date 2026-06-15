"""End-to-end tests using a stub OpenAI-compatible embeddings server.

The stub embeds text as an L2-normalized bag-of-words over a fixed vocab, so
semantic ranking is deterministic and assertable without a real model.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer

import pytest

from turbovec_mcp import store
from turbovec_mcp.config import Config

# turbovec requires dim be a multiple of 8; keep the vocab length aligned.
VOCAB = ["launch", "server", "model", "embedding", "search", "vector", "index",
         "chunk", "config", "sampling", "token", "memory", "cache", "python",
         "rust", "quant"]


def _embed(text: str) -> list[float]:
    t = text.lower()
    v = [float(len(re.findall(r"\b" + w + r"\b", t))) for w in VOCAB]
    n = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / n for x in v]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inp = body["input"]
        inp = inp if isinstance(inp, list) else [inp]
        data = [{"index": i, "embedding": _embed(t)} for i, t in enumerate(inp)]
        out = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture(scope="module")
def endpoint():
    TCPServer.allow_reuse_address = True
    srv = TCPServer(("127.0.0.1", 0), _Handler)  # port 0 = OS picks a free port
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


@pytest.fixture
def cfg(endpoint):
    return Config(endpoint=endpoint, window=8, overlap=2)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "a.py").write_text(
        "def launch_server():\n    # start the model server\n    return 'serving'\n"
    )
    (tmp_path / "b.py").write_text(
        "def build_index():\n    # vector embedding index\n    chunk = 1\n    return chunk\n"
    )
    (tmp_path / "c.md").write_text("# notes\nsampling token cache memory config\n")
    return tmp_path


def test_status_before_index(repo):
    assert store.status(repo)["indexed"] is False


def test_build_and_status(repo, cfg):
    res = store.build(repo, cfg)
    assert res["indexed_chunks"] == 3
    assert res["dim"] == len(VOCAB)
    st = store.status(repo)
    assert st["indexed"] is True
    assert st["count"] == 3


def test_search_ranking(repo, cfg):
    store.build(repo, cfg)
    top = store.search(repo, "launch the model server", k=3, cfg=cfg)[0]
    assert top["path"] == "a.py"
    assert top["start_line"] == 1
    assert "launch_server" in top["signature"]  # signature line, not full body
    top2 = store.search(repo, "vector embedding index", k=3, cfg=cfg)[0]
    assert top2["path"] == "b.py"


def test_search_is_terse(repo, cfg):
    store.build(repo, cfg)
    hit = store.search(repo, "launch", k=1, cfg=cfg)[0]
    assert "text" not in hit  # no bodies in search
    assert {"path", "score", "signature", "start_line", "end_line"} <= hit.keys()


def test_fetch_returns_full_source(repo, cfg):
    store.build(repo, cfg)
    hit = store.search(repo, "launch the model server", k=1, cfg=cfg)[0]
    loc = f"{hit['path']}:{hit['start_line']}-{hit['end_line']}"
    got = store.fetch(repo, [loc])[0]
    assert got["path"] == "a.py"
    assert "launch_server" in got["text"]  # full body on demand


def test_fetch_bad_location(repo, cfg):
    store.build(repo, cfg)
    assert "error" in store.fetch(repo, ["not-a-location"])[0]


def test_k_clamped_to_count(repo, cfg):
    store.build(repo, cfg)
    hits = store.search(repo, "anything", k=999, cfg=cfg)
    assert len(hits) == 3  # only 3 chunks exist


def test_search_without_index_errors(repo, cfg):
    with pytest.raises(ValueError):
        store.search(repo, "x", k=1, cfg=cfg)


def test_build_empty_dir_errors(tmp_path, cfg):
    with pytest.raises(ValueError):
        store.build(tmp_path, cfg)


def test_reindex_reflects_changes(repo, cfg):
    store.build(repo, cfg)
    # add a new highly-relevant file, rebuild, confirm it can surface
    (repo / "d.py").write_text("def cache_memory():\n    return 'token cache memory'\n")
    store.build(repo, cfg)
    assert store.status(repo)["count"] == 4
    paths = {h["path"] for h in store.search(repo, "cache memory token", k=4, cfg=cfg)}
    assert "d.py" in paths


# --- AST chunking ---
from pathlib import Path as _P
from turbovec_mcp import chunker, astchunker
from turbovec_mcp.config import Config as _Cfg


def test_ast_chunks_on_function_boundaries(tmp_path):
    src = (
        "import os\n"                       # 1
        "\n"                                # 2
        "def alpha(x):\n"                   # 3
        "    return x + 1\n"                # 4
        "\n"                                # 5
        "def beta(y):\n"                    # 6
        "    return y * 2\n"                # 7
    )
    (tmp_path / "m.py").write_text(src)
    cfg = _Cfg()
    chunks = list(chunker.iter_chunks(tmp_path, cfg))
    spans = {(c.start_line, c.end_line) for c in chunks}
    # alpha and beta each become a whole-definition chunk on clean boundaries
    assert (3, 4) in spans
    assert (6, 7) in spans
    # the import line is still covered (gap fill)
    assert any(c.start_line == 1 for c in chunks)


def test_ast_godclass_splits_into_methods(tmp_path):
    # a class too big to embed whole -> methods become individual chunks
    methods = "".join(
        f"    def m{i}(self):\n        return {i}\n" for i in range(40)
    )
    (tmp_path / "big.py").write_text("class God:\n" + methods)
    cfg = _Cfg(max_embed_chars=200)  # force the class to exceed the cap
    chunks = list(chunker.iter_chunks(tmp_path, cfg))
    # many small method chunks, not one giant class chunk
    assert len(chunks) >= 20
    assert max(c.end_line - c.start_line for c in chunks) < 30


def test_unsupported_ext_falls_back(tmp_path):
    (tmp_path / "notes.txt").write_text("\n".join(f"line {i}" for i in range(30)))
    cfg = _Cfg(window=8, overlap=2)
    chunks = list(chunker.iter_chunks(tmp_path, cfg))
    assert len(chunks) >= 1  # line-window fallback still produces chunks


def test_embed_shrinks_on_500(tmp_path):
    """A server that rejects long inputs must not abort the run."""
    import json as _json, threading as _th
    from http.server import BaseHTTPRequestHandler as _BH
    from socketserver import TCPServer as _TS
    from turbovec_mcp.embedder import embed as _embed

    LIMIT = 400  # chars; longer single inputs 500 (shrink floors at 256, so reachable)

    class H(_BH):
        def log_message(self, *a): pass
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            inp = body["input"]; inp = inp if isinstance(inp, list) else [inp]
            if any(len(x) > LIMIT for x in inp):
                self.send_response(500); self.end_headers(); self.wfile.write(b"too long"); return
            data = [{"index": i, "embedding": [0.0] * 16} for i, _ in enumerate(inp)]
            out = _json.dumps({"data": data}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
            self.wfile.write(out)

    _TS.allow_reuse_address = True
    srv = _TS(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = _Cfg(endpoint=f"http://127.0.0.1:{port}/v1", batch_size=4, max_embed_chars=10000)
        texts = ["short", "x" * 500, "y" * 2000, "ok"]  # two exceed LIMIT
        vecs = _embed(texts, c)
        assert vecs.shape == (4, 16)  # all four returned despite 500s
    finally:
        srv.shutdown()


# --- file selection (no embedding server; pure filesystem) ---

def test_seed_config_from_nested_gitignore(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".gitignore").write_text("build/\n")
    cp = chunker.seed_config(tmp_path, _Cfg())
    assert cp == chunker.config_path(tmp_path)
    data = json.loads(cp.read_text())
    exclude = data["exclude"]
    # nested gitignore "build/" anchored to sub/ -> workspace-relative globs
    assert any(g.startswith("sub/") and "build" in g for g in exclude)
    # built-in default skips are baked in too
    assert "**/node_modules/**" in exclude


def test_list_files_default_walk_skips_dep_dirs(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("y = 2\n")
    files = {p.name for p in chunker.list_files(tmp_path, _Cfg())}
    assert "real.py" in files
    assert "x.py" not in files


def test_list_files_honors_config_exclude(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "v.py").write_text("a = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("b = 2\n")
    cp = chunker.config_path(tmp_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"include": [], "exclude": ["vendor/**"]}))
    rels = {p.relative_to(tmp_path).as_posix() for p in chunker.list_files(tmp_path, _Cfg())}
    assert "src/a.py" in rels
    assert "vendor/v.py" not in rels


def test_list_files_honors_config_include_whitelist(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.py").write_text("a = 1\n")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "b.py").write_text("b = 2\n")
    cp = chunker.config_path(tmp_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"include": ["keep/**"], "exclude": []}))
    rels = {p.relative_to(tmp_path).as_posix() for p in chunker.list_files(tmp_path, _Cfg())}
    assert rels == {"keep/a.py"}


def test_seed_config_force_semantics(tmp_path):
    first = chunker.seed_config(tmp_path, _Cfg())
    assert first is not None
    # exists + force=False -> None, file untouched
    assert chunker.seed_config(tmp_path, _Cfg(), force=False) is None
    # force=True -> overwrites, returns path
    assert chunker.seed_config(tmp_path, _Cfg(), force=True) == chunker.config_path(tmp_path)
