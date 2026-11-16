"""StudyMind — cross-topic, summarize and connect run on our retriever + local LLM.

A fake retriever records (query, topic_id, k) so tests can check that the
retrieval query is the short question/topic, never the generation instruction.
"""

import pytest

from core import concepts, query, retrieval, topics
from core.db import Chunk, ConceptEdge, ConceptExtraction, Source, session
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
        monkeypatch.setattr(query, "get_retriever", lambda rewrite=True: r)
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
async def test_cross_topic_no_material_skips_llm(fake, monkeypatch):
    _, prompts = fake({})
    monkeypatch.setattr(query, "_topics_with_chunks", lambda ids: set())
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
async def test_features_report_llm_error(fake):
    fake({"ml": [_hit("Notes", "x", 1)], None: [_hit("Notes", "x", 1)]}, reply=RuntimeError("ollama down"))
    assert "RuntimeError" in (await query.cross_topic_ask("q", "ml", ["ml"]))["error"]
    assert "RuntimeError" in (await query.summarize_topic("x"))["error"]


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


@pytest.mark.asyncio
async def test_cross_topic_interleaves_sub_questions_and_dedupes(fake):
    first = [_hit("Notes", f"a{i}", i) for i in range(1, 7)]
    second = [_hit("Notes", "b", 7), first[0]]  # distinct hit, plus a repeat of chunk 1
    fake({("What is A?", "ml"): first, ("What is B?", "ml"): second},
         reply=["1. What is A?\n2. What is B?", "ok"])
    res = await query.cross_topic_ask("A and B?", "ml", ["ml"])
    ids = [c["chunk_id"] for c in res["citations"]]
    assert ids == [1, 7, 2, 3, 4, 5]  # chunk 7 survives the cap; chunk 1 appears once


def _planted(n, edges_by_chunk, done=None):
    """n real chunks (ids returned) with planted concept edges {index: [(h, r, t)]}."""
    tid = topics.create_topic("Connect Topic")["id"]
    with session() as s:
        src = Source(topic_id=tid, source_type="note", title="Notes")
        s.add(src)
        s.commit()
        chunks = [Chunk(source_id=src.id, topic_id=tid, chunker="x", ord=i, text=f"text {i}") for i in range(n)]
        s.add_all(chunks)
        s.commit()
        ids = [c.id for c in chunks]
        for i, triples in edges_by_chunk.items():
            for h, r, t in triples:
                s.add(ConceptEdge(chunk_id=ids[i], source_id=src.id, topic_id=tid, head=h, relation=r, tail=t,
                                  head_norm=concepts.normalise(h), tail_norm=concepts.normalise(t)))
        for i in (range(n) if done is None else done):
            s.add(ConceptExtraction(chunk_id=ids[i]))
        s.commit()
    return ids


def test_shared_concepts_overlap_and_rank():
    edges = {i: [("Alpha", "is", f"x{i}")] for i in range(3)}  # alpha on 3 chunks
    edges[3] = [("Beta", "is", "y")]
    edges[4] = [("Beta", "is", "z")]
    for k in range(10):  # ten more concepts, each on 2 chunks, one per side
        edges.setdefault(5 + 2 * k, []).append((f"c{k:02}", "r", "p"))
        edges.setdefault(6 + 2 * k, []).append((f"c{k:02}", "r", "q"))
    ids = _planted(26, edges)
    side_a = [ids[0], ids[1], ids[3]] + [ids[5 + 2 * k] for k in range(10)]
    side_b = [ids[2], ids[4]] + [ids[6 + 2 * k] for k in range(10)]
    out = concepts.shared_concepts(side_a, side_b, set())
    assert len(out) == concepts.TOP_SHARED
    assert [o.name for o in out[:2]] == ["Alpha", "Beta"] and out[0].support == 3
    assert [o.name for o in out[2:]] == ["c00", "c01", "c02", "c03", "c04", "c05"]


def test_topic_names_excluded():
    ids = _planted(2, {0: [("Neural Networks", "use", "Gradients")], 1: [("neural network", "learn", "Gradients")]})
    out = concepts.shared_concepts([ids[0]], [ids[1]], {concepts.normalise("Neural Networks")})
    assert [o.name for o in out] == ["Gradients"]


def test_overlap_chunk_alone_shares_nothing():
    ids = _planted(3, {0: [("Alpha", "r", "Beta")], 1: [("Alpha", "r", "Gamma")]})
    assert concepts.shared_concepts([ids[0]], [ids[0]], set()) == []
    out = concepts.shared_concepts([ids[0]], [ids[0], ids[1]], set())  # Alpha also in distinct chunk 1
    assert [o.name for o in out] == ["Alpha"] and out[0].edge_a.chunk_id != out[0].edge_b.chunk_id


def _two_topics(fake, ids, reply="never"):
    hit = lambda cid: Hit(cid, 1, "Notes", 4, None, f"text {cid}", 1.0)  # noqa: E731
    return fake({("a", None): [hit(ids[0]), hit(ids[1])], ("b", None): [hit(ids[2]), hit(ids[3])]}, reply=reply)


@pytest.mark.asyncio
async def test_connect_ignores_irrelevant_hits(fake, monkeypatch):
    # Same overlap as test_connect_builds_graph_without_llm, but topic b's hits score below the cutoff
    monkeypatch.setattr(query.settings, "retriever", "hybrid")
    ids = _planted(4, {0: [("Gradient", "drives", "Learning")], 2: [("Optimiser", "follows", "gradient")]})
    r, _ = _two_topics(fake, ids)
    for h in r.hits[("b", None)]:
        h.score = -6.0
    res = await query.connect_topics("a", "b")
    assert res["connection_method"] == "none" and res["nodes"] == []
    assert "No connections found between 'a' and 'b'" in res["summary"]


@pytest.mark.asyncio
async def test_connect_builds_graph_without_llm(fake):
    ids = _planted(4, {0: [("Gradient", "drives", "Learning")], 2: [("Optimiser", "follows", "gradient")]})
    _, prompts = _two_topics(fake, ids)
    res = await query.connect_topics("a", "b")
    assert res["connection_method"] == "concept_overlap" and res["error"] is None and prompts == []
    assert [n["label"] for n in res["nodes"] if n["source_type"] == "shared"] == ["Gradient"]
    assert {e["relationship"] for e in res["edges"]} == {"drives", "follows"}
    assert "Notes · p.4" in res["summary"] and "still being processed" not in res["summary"]


@pytest.mark.asyncio
async def test_connect_pending_note(fake):
    ids = _planted(4, {0: [("Gradient", "drives", "Learning")], 2: [("Gradient", "follows", "x")]}, done=[0, 1, 2])
    _two_topics(fake, ids)
    res = await query.connect_topics("a", "b")
    assert "1 of 4 passages are still being processed — try again shortly." in res["summary"]


@pytest.mark.asyncio
async def test_connect_no_overlap_empty_state(fake):
    ids = _planted(4, {0: [("Gradient", "drives", "Learning")], 2: [("Enzyme", "binds", "Substrate")]})
    _two_topics(fake, ids)
    res = await query.connect_topics("a", "b")
    assert res["connection_method"] == "none" and res["nodes"] == []
    assert "No connections found between 'a' and 'b'" in res["summary"]


def test_context_labels_media_and_citations_carry_it():
    hits = [
        Hit(1, 1, "Slides", 4, None, "Figure (p.4): cycle", 1.0,
            {"kind": "figure", "image_path": "a.png", "start_s": None, "url": None}),
        Hit(2, 1, "Slides", 5, None, "| a |", 1.0, {"kind": "table", "image_path": None, "start_s": None, "url": None}),
        Hit(3, 2, "YouTube: abc", None, None, "spoken", 1.0,
            {"kind": "video", "image_path": None, "start_s": 754.0, "url": "https://youtu.be/abc?t=754"}),
        Hit(4, 1, "Slides", 6, None, "plain", 1.0),
    ]
    text, cites = query._context(hits)
    headers = [line.split(">", 1)[1] for line in text.splitlines() if line.startswith("[")]
    assert headers == ["Slides, p.4 — Figure", "Slides, p.5 — Table", "YouTube: abc @ 12:34", "Slides, p.6"]
    assert [(c["kind"], c["image_url"], c["url"]) for c in cites] == [
        ("figure", "/figures/1", None), ("table", None, None), ("video", None, "https://youtu.be/abc?t=754"),
        ("text", None, None),
    ]


@pytest.mark.asyncio
async def test_summarize_unrelated_topic_skips_llm(fake, monkeypatch):
    monkeypatch.setattr(query.settings, "retriever", "hybrid")
    _, prompts = fake({None: [Hit(1, 1, "Bio", 1, None, "enzymes speed reactions", -9.0)]}, reply="Made up [1]")
    res = await query.summarize_topic("French Revolution")
    assert prompts == [] and "No information found" in res["summary"]


@pytest.fixture
def counted(monkeypatch):
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        return "An answer [1]"

    monkeypatch.setattr(llm, "complete", complete)
    return calls


async def _ingest(text="Transformers use attention over tokens.", title="n"):
    from core import ingest
    return await ingest.ingest_text(text, title)


@pytest.mark.asyncio
async def test_repeated_question_is_answered_from_cache(counted):
    await _ingest()
    a = await query.ask("What is attention?")
    n = len(counted)
    b = await query.ask("  what IS   attention? ")
    assert len(counted) == n and a == b and a["citations"]
    b["answer"] = "mutated"
    assert (await query.ask("what is attention?"))["answer"] == a["answer"]


@pytest.mark.asyncio
async def test_cache_cleared_by_ingest(counted):
    await _ingest()
    await query.ask("What is attention?")
    n = len(counted)
    await _ingest("Attention weights are softmax scores.", "m")
    await query.ask("What is attention?")
    assert len(counted) == n + 1


@pytest.mark.asyncio
async def test_cache_cleared_by_delete_and_move(counted):
    src = (await _ingest())["source_id"]
    topic = topics.create_topic("Other")["id"]
    await query.ask("What is attention?")
    n = len(counted)
    assert topics.move_source(src, topic)
    await query.ask("What is attention?")
    assert len(counted) == n + 1
    assert topics.delete_source(src)
    res = await query.ask("What is attention?")
    assert res["citations"] == []  # a stale cached answer would still cite the deleted source
    assert len(counted) == n + 1  # nothing left: no LLM call


@pytest.mark.asyncio
async def test_cache_drops_old_answers_once_material_changes(counted):
    src = (await _ingest())["source_id"]
    await query.ask("What is attention?")
    topics.delete_source(src)
    await _ingest()
    await query.ask("Something else?")
    assert all(k[-1] == retrieval.corpus_version() for k in query._cache)  # no answer citing deleted text


@pytest.mark.asyncio
async def test_cache_evicts_oldest(counted, monkeypatch):
    monkeypatch.setattr(query, "CACHE_SIZE", 2)
    await _ingest()
    for q in ("one?", "two?", "three?"):
        await query.ask(q)
    n = len(counted)
    await query.ask("one?")
    assert len(counted) == n + 1


@pytest.mark.asyncio
async def test_errors_and_no_material_are_not_cached(counted, monkeypatch):
    assert (await query.ask("What is attention?"))["citations"] == []  # empty corpus
    await _ingest()
    assert (await query.ask("What is attention?"))["citations"]  # retried, now cached
    n = len(counted)
    await query.ask("What is attention?")
    assert len(counted) == n

    async def boom(*a, **k):
        raise RuntimeError("down")

    good = llm.complete
    monkeypatch.setattr(llm, "complete", boom)
    assert (await query.ask("Another question?"))["error"]
    monkeypatch.setattr(llm, "complete", good)
    assert (await query.ask("Another question?"))["error"] is None


@pytest.mark.asyncio
async def test_cache_key_includes_settings(counted, monkeypatch):
    await _ingest()
    await query.ask("What is attention?")
    n = len(counted)
    monkeypatch.setattr(query.settings, "rerank", not query.settings.rerank)
    await query.ask("What is attention?")
    assert len(counted) == n + 1


async def _events(question, topic_id=None):
    return [e async for e in query.ask_stream(question, topic_id)]


@pytest.mark.asyncio
async def test_ask_stream_yields_citations_tokens_done(counted, monkeypatch):
    await _ingest()

    async def stream(prompt, system=None, **kw):
        for t in ("An ", "answer ", "[1]"):
            yield t

    monkeypatch.setattr(llm, "stream", stream)
    ev = await _events("What is attention?")
    names = [n for n, _ in ev]
    assert names == ["citations", "token", "token", "token", "done"]
    assert ev[-1][1]["answer"] == "".join(p for n, p in ev if n == "token")
    assert ev[-1][1]["citations"] == ev[0][1] and ev[0][1]


@pytest.mark.asyncio
async def test_ask_stream_hits_cache_after_ask(counted):
    await _ingest()
    a = await query.ask("What is attention?")
    n = len(counted)
    ev = await _events("What is attention?")
    assert len(counted) == n
    assert [x for x, _ in ev] == ["citations", "token", "done"] and ev[-1][1] == a


@pytest.mark.asyncio
async def test_ask_stream_stores_for_ask(counted):
    await _ingest()
    await _events("What is attention?")
    n = len(counted)
    await query.ask("What is attention?")
    assert len(counted) == n


@pytest.mark.asyncio
async def test_ask_stream_error_midstream_caches_nothing(counted, monkeypatch):
    await _ingest()
    good = llm.stream

    async def broken(prompt, system=None, **kw):
        yield "partial "
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "stream", broken)
    ev = await _events("What is attention?")
    assert [n for n, _ in ev][-1] == "error" and "down" in ev[-1][1]
    monkeypatch.setattr(llm, "stream", good)
    n = len(counted)
    await query.ask("What is attention?")
    assert len(counted) == n + 1


@pytest.mark.asyncio
async def test_ask_stream_no_material(counted):
    ev = await _events("What is attention?")
    assert [n for n, _ in ev] == ["done"] and "No results found" in ev[0][1]["answer"] and counted == []
