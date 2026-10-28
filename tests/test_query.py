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
        reply = state["reply"]
        if isinstance(reply, list):  # consumed in order, one per LLM call
            reply = reply.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(query, "_topics_with_chunks", lambda ids: set(ids))

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
    assert len(prompts) == 2 and "gradient descent steps downhill" in prompts[-1] and "enzymes" in prompts[-1]
    assert (res["primary_section"], res["cross_section"]) == ("steps downhill [1]", "enzymes [2]")
    assert [c["source_title"] for c in res["citations"]] == ["Notes", "Bio"]
    assert res["error"] is None


@pytest.mark.asyncio
async def test_cross_topic_without_other_topics(fake):
    r, _ = fake({"ml": [_hit("Notes", "gradient descent", 1)]}, reply=["x", "An answer [1]"])
    res = await query.cross_topic_ask("q", "ml", ["ml"])
    assert r.calls == [("q", "ml", 6)]
    assert res["primary_section"] == "An answer [1]" and res["cross_section"] == ""


@pytest.mark.asyncio
async def test_cross_topic_no_material_skips_llm(fake):
    _, prompts = fake({})
    res = await query.cross_topic_ask("q", "ml", ["ml", "bio"])
    assert len(prompts) == 1 and "No results found" in res["answer"]  # decomposition only


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


@pytest.mark.asyncio
async def test_connect_needs_passages_for_both_topics(fake):
    # Only one side has material: anything the LLM "connects" would be invented
    _, prompts = fake({("optimisation", None): [_hit("Notes", "gradient descent", 1)]},
                      reply="CONCEPT: Gradient | REASON: r")
    res = await query.connect_topics("optimisation", "photosynthesis")
    assert prompts == [] and res["connection_method"] == "none" and res["nodes"] == []


@pytest.mark.asyncio
async def test_connect_uses_llm_even_when_passages_overlap(fake):
    # Same short-line chunk for both topics must not become a "shared concept" by itself
    shared = [_hit("Web", "Source: Web Article\nURL: https://x.test", 1)]
    _, prompts = fake({("transformers", None): shared, ("attention", None): shared},
                      reply="CONCEPT: Self-attention | REASON: r")
    res = await query.connect_topics("transformers", "attention")
    assert res["connection_method"] == "ai_synthesized" and len(prompts) == 1
    assert [n["label"] for n in res["nodes"] if n["source_type"] == "shared"] == ["Self-attention"]
    assert prompts[0].count("URL: https://x.test") == 1  # duplicate chunk sent once


def test_concept_parser_handles_numbered_and_bold_lines():
    reply = (
        "1. CONCEPT: Gradient | REASON: both step downhill\n"
        "- **CONCEPT:** Fitness landscape | **REASON:** both climb\n"
        "CONCEPT: Hill climbing – REASON: local search\n"
    )
    nodes, _ = query._parse_concept_lines(reply, "a", "b")
    assert [n["label"] for n in nodes if n["source_type"] == "shared"] == [
        "Gradient", "Fitness landscape", "Hill climbing",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    "**From ml**: steps downhill [1]\n\n**Connected from other topics**: enzymes [2]",
    "### From ml\nsteps downhill [1]\n\n### Connected from other topics\nenzymes [2]",
    "From ml: steps downhill [1]\n\nConnected from other topics: enzymes [2]",
])
async def test_cross_topic_split_tolerates_heading_styles(fake, reply):
    fake({"ml": [_hit("Notes", "gradient descent", 1)], "bio": [_hit("Bio", "enzymes", 2)]}, reply=["x", reply])
    res = await query.cross_topic_ask("q", "ml", ["ml", "bio"])
    assert (res["primary_section"], res["cross_section"]) == ("steps downhill [1]", "enzymes [2]")


@pytest.mark.asyncio
async def test_connect_llm_can_decline(fake):
    # Retrieval always returns the nearest chunks, even for a topic the material never covers
    _, prompts = fake({("optimisation", None): [_hit("Notes", "gradient descent", 1)],
                       ("medieval poetry", None): [_hit("Bio", "natural selection", 2)]}, reply="NONE")
    res = await query.connect_topics("optimisation", "medieval poetry")
    assert res["connection_method"] == "none" and res["nodes"] == []
    assert "NONE" in prompts[0]


@pytest.mark.asyncio
async def test_cross_topic_decomposes_and_retrieves_each_sub_question(fake):
    r, prompts = fake(
        {"ml": [_hit("Notes", "aaa", 1)], "bio": [_hit("Bio", "bbb", 2)]},
        reply=["1. What is A?\n2. What is B?", "synthesis"],
    )
    await query.cross_topic_ask("A and B?", "ml", ["ml", "bio"])
    assert sorted(r.calls) == sorted(
        (q, t, k) for q in ("What is A?", "What is B?") for t, k in (("ml", 6), ("bio", 2))
    )
    assert "Question: A and B?" in prompts[-1]


@pytest.mark.asyncio
async def test_cross_topic_skips_topics_without_chunks(fake, monkeypatch):
    r, _ = fake({"ml": [_hit("Notes", "aaa", 1)]}, reply=["1. What is A?\n2. What is B?", "ok"])
    monkeypatch.setattr(query, "_topics_with_chunks", lambda ids: {"ml"})
    await query.cross_topic_ask("q?", "ml", ["ml", "bio"])
    assert {t for _, t, _ in r.calls} == {"ml"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, expected", [
    ("**1. What is A?**\n\nWhat is A?\nnot a question", ["Q?"]),
    ("- What is A?\n- What is B?\n* What is C?\n3) What is D?\nWhat is A?",
     ["What is A?", "What is B?", "What is C?"]),
    (RuntimeError("down"), ["Q?"]),
])
async def test_sub_questions_cleanup(fake, reply, expected):
    fake({}, reply=reply)
    assert await query._sub_questions("Q?") == expected


@pytest.mark.asyncio
async def test_topics_with_chunks():
    from core.ingest import ingest_text

    await ingest_text("Gradient descent steps downhill", title="Opt", topic_id="general")
    assert query._topics_with_chunks(["general", "nope"]) == {"general"}
    assert query._topics_with_chunks([]) == set()
