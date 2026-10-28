"""
Shared fixtures: every test runs offline against a temp SQLite DB, a mocked
Ollama (httpx.MockTransport), fake embedders/reranker and a fake LLM.
"""

import re
import sys
import zlib
from pathlib import Path

import httpx
import pytest
from qdrant_client import models

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import chunking, config, db, retrieval, topics  # noqa: E402
from core.llm import llm  # noqa: E402

def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def fake_embed_dense(texts, *, query):
    """Hashed bag of words, plus a constant dim so no vector is zero."""
    dim = retrieval.EMBED_MODELS[config.settings.embed_model][1]
    out = []
    for t in texts:
        v = [0.0] * dim
        v[0] = 1.0
        for w in _words(t):
            v[1 + zlib.crc32(w.encode()) % (dim - 1)] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        out.append([x / norm for x in v])
    return out


def fake_embed_sparse(texts, *, query):
    out = []
    for t in texts:
        counts: dict[int, float] = {}
        for w in _words(t):
            i = zlib.crc32(w.encode()) % 2**20
            counts[i] = counts.get(i, 0.0) + 1.0
        idx = sorted(counts)
        out.append(models.SparseVector(indices=idx, values=[counts[i] for i in idx]))
    return out


def fake_rerank(query, texts):
    q = set(_words(query))
    return [float(len(q & set(_words(t)))) for t in texts]


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """Returns the list of mocked HTTP paths called, in order."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        # Guard: anything else would be a real network call
        raise AssertionError(f"unexpected network call: {request.url}")

    monkeypatch.setattr(config.settings, "llm_model", "ollama/qwen2.5:7b")
    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(config.settings, "langfuse_public_key", "")
    monkeypatch.setattr(config.settings, "qdrant_path", ":memory:")
    monkeypatch.setattr(config.settings, "embed_model", "bge-small")
    monkeypatch.setattr(retrieval, "embed_dense", fake_embed_dense)
    monkeypatch.setattr(retrieval, "embed_sparse", fake_embed_sparse)
    monkeypatch.setattr(retrieval, "rerank", fake_rerank)
    monkeypatch.setattr(topics, "LEGACY_TOPICS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(topics, "AUDIO_DIR", tmp_path)

    async def fake_complete(prompt, system=None, **kwargs):
        return "Transformer Models"

    monkeypatch.setattr(llm, "complete", fake_complete)
    # Whitespace tokens instead of the bge tokenizer (no download)
    monkeypatch.setattr(chunking, "token_spans", lambda t: [m.span() for m in re.finditer(r"\S+", t)])

    db.reset_engine()
    retrieval.reset_store()
    config.set_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    topics.init_db()
    yield calls
    db.reset_engine()
    retrieval.reset_store()
    config.set_http_client(None)
