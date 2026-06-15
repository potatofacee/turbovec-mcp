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
    assert "launch_server" in top["text"]  # text re-read from file
    top2 = store.search(repo, "vector embedding index", k=3, cfg=cfg)[0]
    assert top2["path"] == "b.py"


def test_compact_omits_text(repo, cfg):
    store.build(repo, cfg)
    hit = store.search(repo, "launch", k=1, cfg=cfg, compact=True)[0]
    assert "text" not in hit
    assert "path" in hit and "score" in hit


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
