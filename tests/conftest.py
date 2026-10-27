"""
Shared fixtures: every test runs offline against a temp SQLite DB, a mocked
Cognee Cloud / Ollama (httpx.MockTransport) and a fake LLM.
"""

import sys
from pathlib import Path

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import config, db, topics  # noqa: E402
from core.llm import llm  # noqa: E402

RECALL_TEXT = "Self-attention lets each token attend to every other token."


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """Returns the list of mocked HTTP paths called, in order."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/api/v1/remember":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/api/v1/recall":
            return httpx.Response(200, json=[{"text": RECALL_TEXT, "dataset_name": "transformers"}])
        if path == "/api/v1/datasets/":
            return httpx.Response(200, json=[])
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        # Guard: anything else would be a real network call
        raise AssertionError(f"unexpected network call: {request.url}")

    monkeypatch.setattr(config.settings, "cognee_base_url", "https://cognee.test")
    monkeypatch.setattr(config.settings, "cognee_api_key", "test-key")
    monkeypatch.setattr(config.settings, "llm_model", "ollama/qwen2.5:7b")
    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(config.settings, "langfuse_public_key", "")
    monkeypatch.setattr(topics, "LEGACY_TOPICS_FILE", tmp_path / "topics.json")
    monkeypatch.setattr(topics, "AUDIO_DIR", tmp_path)

    async def fake_complete(prompt, system=None, **kwargs):
        return "Transformer Models"

    monkeypatch.setattr(llm, "complete", fake_complete)

    db.reset_engine()
    config.set_http_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    topics.init_db()
    yield calls
    db.reset_engine()
    config.set_http_client(None)
