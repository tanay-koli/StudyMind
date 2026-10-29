"""StudyMind — multi-query and HyDE rewriting."""

import pytest

from core import retrieval, rewrite
from core.config import settings
from core.ingest import ingest_text
from core.llm import llm
from core.retrieval import Hit


def hit(i: int) -> Hit:
    return Hit(i, 1, "T", None, None, f"chunk {i}", 1.0)


class RecordingRetriever:
    def __init__(self, results: dict[str, list[int]] | None = None):
        self.results = results or {}
        self.calls: list[tuple] = []

    async def retrieve(self, query, topic_id, k=6, rerank_query=None):
        self.calls.append((query, topic_id, k, rerank_query))
        return [hit(i) for i in self.results.get(query, [])]


def reply(monkeypatch, text=None, exc=None):
    async def fake(prompt, system=None, **kw):
        if exc:
            raise exc
        return text

    monkeypatch.setattr(llm, "complete", fake)


@pytest.mark.asyncio
async def test_multi_fuses_original_and_paraphrases(monkeypatch):
    reply(monkeypatch, "1. what is X?\n\n- What is X?\nexplain Y\nexplain Z\nextra")
    base = RecordingRetriever({"define X": [1, 2], "what is X?": [3, 2], "explain Y": [4, 2], "explain Z": [5]})

    res = await rewrite.RewritingRetriever(base, "multi").retrieve("define X", "t", k=3)

    assert [c[0] for c in base.calls] == ["define X", "what is X?", "explain Y", "explain Z"]
    assert all(c[2] == 3 for c in base.calls)
    assert res[0].chunk_id == 2 and len(res) <= 3


@pytest.mark.asyncio
async def test_multi_drops_original_echo(monkeypatch):
    reply(monkeypatch, "DEFINE x\nexplain Y")
    base = RecordingRetriever()

    await rewrite.RewritingRetriever(base, "multi").retrieve("define X", None)

    assert [c[0] for c in base.calls] == ["define X", "explain Y"]


@pytest.mark.asyncio
async def test_hyde_retrieves_with_answer_reranks_with_question(monkeypatch):
    reply(monkeypatch, "X is a thing.")
    base = RecordingRetriever()

    await rewrite.RewritingRetriever(base, "hyde").retrieve("what is X?", "t", k=4)

    assert base.calls == [("X is a thing.", "t", 4, "what is X?")]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["multi", "hyde"])
async def test_rewrite_falls_back_when_llm_fails(monkeypatch, mode):
    reply(monkeypatch, exc=RuntimeError("down"))
    base = RecordingRetriever()

    await rewrite.RewritingRetriever(base, mode).retrieve("what is X?", "t")

    assert base.calls == [("what is X?", "t", 6, None)]


@pytest.mark.asyncio
async def test_hybrid_reranks_with_rerank_query():
    await ingest_text("alpha shared", title="A", topic_id="general")
    await ingest_text("beta shared", title="B", topic_id="general")
    r = retrieval.HybridRetriever()

    by_alpha = await r.retrieve("shared", "general", rerank_query="alpha")
    by_beta = await r.retrieve("shared", "general", rerank_query="beta")

    assert by_alpha[0].source_title == "A"
    assert by_beta[0].source_title == "B"


def test_get_retriever_wraps_when_flag_set(monkeypatch):
    assert not isinstance(retrieval.get_retriever(), rewrite.RewritingRetriever)
    monkeypatch.setattr(settings, "query_rewrite", "hyde")
    r = retrieval.get_retriever()
    assert isinstance(r, rewrite.RewritingRetriever) and r.mode == "hyde"
