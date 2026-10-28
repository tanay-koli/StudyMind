"""StudyMind — cross-topic, summarize and connect run on our retriever + local LLM.

A fake retriever records (query, topic_id, k) so tests can check that the
retrieval query is the short question/topic, never the generation instruction.
"""

import pytest

from core import query
from core.llm import llm
from core.retrieval import Hit


class FakeRetriever:
    def __init__(self, hits_by_topic):
        self.hits = hits_by_topic
        self.calls = []

    async def retrieve(self, q, topic_id, k=6):
        self.calls.append((q, topic_id, k))
        return self.hits.get((q, topic_id), self.hits.get(topic_id, []))[:k]


def _hit(title, text, cid):
    return Hit(cid, cid, title, 1, None, text, 1.0)


@pytest.fixture
def fake(monkeypatch):
    """Install a fake retriever; returns (retriever, prompts, set_reply)."""
    state = {"reply": "ok", "prompts": [], "retriever": None}

    async def complete(prompt, system=None, **kw):
        state["prompts"].append(prompt)
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return state["reply"]

    monkeypatch.setattr(llm, "complete", complete)

    def install(hits_by_topic, reply="ok"):
        r = FakeRetriever(hits_by_topic)
        state["reply"] = reply
        monkeypatch.setattr(query, "get_retriever", lambda: r)
        return r, state["prompts"]

    return install


@pytest.mark.asyncio
async def test_cross_topic_retrieves_with_question_per_topic(fake):
    r, prompts = fake(
        {"ml": [_hit("Notes", "gradient descent steps downhill", 1)],
         "bio": [_hit("Bio", "enzymes lower activation energy", 2)]},
        reply="**From ml:** steps downhill [1]\n\n**Connected from other topics:** enzymes [2]",
    )
    res = await query.cross_topic_ask("how do things go downhill?", "ml", ["ml", "bio"])
    assert r.calls == [("how do things go downhill?", "ml", 6), ("how do things go downhill?", "bio", 2)]
    assert len(prompts) == 1 and "gradient descent steps downhill" in prompts[0] and "enzymes" in prompts[0]
    assert (res["primary_section"], res["cross_section"]) == ("steps downhill [1]", "enzymes [2]")
    assert [c["source_title"] for c in res["citations"]] == ["Notes", "Bio"]
    assert res["error"] is None


@pytest.mark.asyncio
async def test_cross_topic_without_other_topics(fake):
    r, _ = fake({"ml": [_hit("Notes", "gradient descent", 1)]}, reply="An answer [1]")
    res = await query.cross_topic_ask("q", "ml", ["ml"])
    assert r.calls == [("q", "ml", 6)]
    assert res["primary_section"] == "An answer [1]" and res["cross_section"] == ""


@pytest.mark.asyncio
async def test_cross_topic_no_material_skips_llm(fake):
    _, prompts = fake({})
    res = await query.cross_topic_ask("q", "ml", ["ml", "bio"])
    assert prompts == [] and "No results found" in res["answer"]


@pytest.mark.asyncio
async def test_summarize_retrieves_topic_text(fake):
    r, prompts = fake({None: [_hit("Notes", "backprop applies the chain rule", 1)]}, reply="Summary [1]")
    res = await query.summarize_topic("backpropagation")
    assert r.calls == [("backpropagation", None, 8)]
    assert "backprop applies the chain rule" in prompts[0]
    assert (res["summary"], res["sources_referenced"]) == ("Summary [1]", ["Notes"])
    assert res["citations"][0]["chunk_id"] == 1


@pytest.mark.asyncio
async def test_connect_falls_back_to_llm_concepts(fake):
    r, prompts = fake(
        {("optimisation", None): [_hit("Notes", "alpha beta", 1)],
         ("evolution", None): [_hit("Bio", "gamma delta", 2)]},
        reply="CONCEPT: Gradient | REASON: both minimise something",
    )
    res = await query.connect_topics("optimisation", "evolution")
    assert ("optimisation", None, 6) in r.calls and ("evolution", None, 6) in r.calls
    assert res["connection_method"] == "ai_synthesized"
    assert "Gradient" in [n["label"] for n in res["nodes"]]
    assert "alpha beta" in prompts[0] and "gamma delta" in prompts[0]


@pytest.mark.asyncio
async def test_features_report_llm_error(fake):
    fake({"ml": [_hit("Notes", "x", 1)], None: [_hit("Notes", "x", 1)]}, reply=RuntimeError("ollama down"))
    assert "RuntimeError" in (await query.cross_topic_ask("q", "ml", ["ml"]))["error"]
    assert "RuntimeError" in (await query.summarize_topic("x"))["error"]
