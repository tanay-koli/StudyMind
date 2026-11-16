"""StudyMind — LLMClient: Ollama context window and the JSON helper."""

import base64
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
async def test_complete_sends_images_as_data_urls(monkeypatch):
    seen = {}

    async def fake_acompletion(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    await LLMClient().complete("describe", images=[b"\x89PNG"], model="ollama/qwen2.5vl:7b")
    content = seen["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
    assert seen["model"] == "ollama/qwen2.5vl:7b" and "images" not in seen


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


@pytest.mark.asyncio
async def test_usage_tally_counts_calls_and_tokens(monkeypatch):
    async def fake_acompletion(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    client = LLMClient()
    await client.complete("hi")
    await client.complete("again")
    assert client.usage == {"calls": 2, "prompt_tokens": 10, "completion_tokens": 4, "cost_usd": 0.0}


def _chunks(*texts, usage=None):
    out = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))], usage=None) for t in texts]
    out.append(SimpleNamespace(choices=[], usage=usage))
    return out


def _fake_stream(seen, chunks):
    async def fake_acompletion(**kw):
        seen.update(kw)

        async def gen():
            for c in chunks:
                yield c
        return gen()
    return fake_acompletion


@pytest.mark.asyncio
async def test_stream_yields_deltas_and_tallies_usage(monkeypatch):
    chunks = _chunks("Hel", "lo", None, usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    monkeypatch.setattr(litellm, "acompletion", _fake_stream({}, chunks))
    client = LLMClient()
    assert [d async for d in client.stream("hi")] == ["Hel", "lo"]
    assert client.usage == {"calls": 1, "prompt_tokens": 5, "completion_tokens": 2, "cost_usd": 0.0}


@pytest.mark.asyncio
async def test_stream_passes_ollama_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(litellm, "acompletion", _fake_stream(seen, _chunks("x")))
    async for _ in LLMClient().stream("hi", model="ollama/qwen2.5:7b"):
        pass
    assert seen["stream"] is True and seen["num_ctx"] == 8192 and seen["api_base"]
    assert seen["stream_options"] == {"include_usage": True}
