"""StudyMind — LLMClient: Ollama context window and the JSON helper."""

from types import SimpleNamespace

import litellm
import pytest

from core.llm import LLMClient


@pytest.mark.asyncio
async def test_complete_sets_num_ctx_for_ollama(monkeypatch):
    seen = {}

    async def fake_acompletion(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))], usage=None)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    await LLMClient().complete("q")
    assert seen["num_ctx"] == 8192
    await LLMClient().complete("q", num_ctx=2048)
    assert seen["num_ctx"] == 2048


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} hope that helps'])
async def test_complete_json_extracts_object(monkeypatch, reply):
    client = LLMClient()

    async def fake_complete(prompt, system=None, **kw):
        assert kw["response_format"] == {"type": "json_object"}
        return reply

    monkeypatch.setattr(client, "complete", fake_complete)
    assert await client.complete_json("q") == {"a": 1}


@pytest.mark.asyncio
async def test_complete_json_raises_on_no_object(monkeypatch):
    client = LLMClient()

    async def fake_complete(prompt, system=None, **kw):
        return "no json here"

    monkeypatch.setattr(client, "complete", fake_complete)
    with pytest.raises(ValueError):
        await client.complete_json("q")
