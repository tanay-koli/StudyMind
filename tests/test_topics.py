"""StudyMind — topic/source storage (SQLite) and cached generation."""

import pytest
from sqlmodel import select

from core import db, query, topics
from core.llm import llm
from core.retrieval import Hit


def test_general_always_exists_and_cannot_be_deleted():
    assert topics.GENERAL_TOPIC_ID in topics.load_topics()
    assert topics.delete_topic(topics.GENERAL_TOPIC_ID) is False


def test_create_topic_dedupes_by_name():
    a = topics.create_topic("Machine Learning")
    b = topics.create_topic("  machine learning ")
    assert a["id"] == b["id"] == "machine_learning"
    assert len(topics.load_topics()) == 2  # General + ML


def test_create_topic_names_with_same_slug_stay_separate():
    ids = [topics.create_topic(n)["id"] for n in ("C", "C++", "C#")]
    assert len(set(ids)) == 3
    assert [topics.load_topics()[i]["name"] for i in ids] == ["C", "C++", "C#"]
    assert topics.create_topic("c++")["id"] == ids[1]


def test_new_source_invalidates_caches():
    tid = topics.create_topic("Physics")["id"]
    topics._update_topic(tid, study_guide="old guide")

    src = topics.add_source(tid, "note", "Newton")
    assert topics.load_topics()[tid]["study_guide"] is None  # new material clears the cache
    assert [s["id"] for s in topics.list_sources(tid)] == [src["id"]]


def test_move_source_between_topics():
    a = topics.create_topic("A")["id"]
    b = topics.create_topic("B")["id"]
    src = topics.add_source(a, "note", "n")
    assert topics.move_source(src["id"], b)["topic_id"] == b
    assert topics.list_sources(a) == []
    assert topics.move_source(src["id"], "missing") is None


def _source_with_concepts(topic_id):
    with db.session() as s:
        src = db.Source(topic_id=topic_id, source_type="note", title="n")
        s.add(src)
        s.commit()
        ch = db.Chunk(source_id=src.id, topic_id=topic_id, chunker="x", ord=0, text="t")
        s.add(ch)
        s.commit()
        s.add(db.ConceptEdge(chunk_id=ch.id, source_id=src.id, topic_id=topic_id, head="a", relation="r",
                             tail="b", head_norm="a", tail_norm="b"))
        s.add(db.ConceptExtraction(chunk_id=ch.id))
        s.commit()
        return src.id


def test_delete_source_purges_concepts():
    sid = _source_with_concepts(topics.GENERAL_TOPIC_ID)
    topics.delete_source(sid)
    with db.session() as s:
        assert s.exec(select(db.ConceptEdge)).first() is None
        assert s.exec(select(db.ConceptExtraction)).first() is None


def test_move_source_moves_concepts():
    b = topics.create_topic("B")["id"]
    sid = _source_with_concepts(topics.GENERAL_TOPIC_ID)
    topics.move_source(sid, b)
    with db.session() as s:
        assert s.exec(select(db.ConceptEdge)).one().topic_id == b


def test_delete_topic_removes_its_sources():
    tid = topics.create_topic("Chemistry")["id"]
    topics.add_source(tid, "pdf", "Atoms")
    assert topics.delete_topic(tid) is True
    assert topics.list_sources() == []


def test_delete_all_sources():
    topics.add_source(topics.GENERAL_TOPIC_ID, "note", "a")
    topics.add_source(topics.GENERAL_TOPIC_ID, "note", "b")
    assert topics.delete_all_sources() == 2
    assert topics.list_sources() == []




@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        return "Generated"

    monkeypatch.setattr(llm, "complete", complete)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("generate, marker", [
    (topics.generate_audio_script, "podcast-style"),
])
async def test_generation_retrieves_topic_name_not_instruction(monkeypatch, llm_calls, generate, marker):
    calls = []

    class Recorder:
        async def retrieve(self, q, topic_id, k=6):
            calls.append((q, topic_id, k))
            return [Hit(1, 1, "Notes", 2, None, "self-attention weighs tokens", 1.0)]

    monkeypatch.setattr(query, "get_retriever", Recorder)
    tid = topics.create_topic("Transformers")["id"]
    topics.add_source(tid, "note", "n")

    assert await generate(tid) == "Generated"
    assert calls == [("Transformers", tid, 8)]
    assert "self-attention weighs tokens" in llm_calls[0] and marker in llm_calls[0]


GUIDE_REPLY = '{"key_points": ["kp"], "terms": [], "overview": "o", "key_concepts": [], "review_questions": []}'


@pytest.fixture
def guide_llm(monkeypatch):
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        return GUIDE_REPLY

    monkeypatch.setattr(llm, "complete", complete)
    return calls


async def _topic_with_text(name="Transformers"):
    from core.ingest import ingest_text

    tid = topics.create_topic(name)["id"]
    await ingest_text("Self-attention weighs every token", title="L1", topic_id=tid)
    return tid


@pytest.mark.asyncio
async def test_study_guide_is_cached(guide_llm):
    from core.studyguide import StudyGuide

    tid = await _topic_with_text()
    first = await topics.generate_study_guide(tid)
    n = len(guide_llm)
    second = await topics.generate_study_guide(tid)
    assert first == second and len(guide_llm) == n  # second call served from the DB
    assert first["data"]["sections"][0]["key_points"] == ["kp"] and "# Study Guide: Transformers" in first["guide"]
    StudyGuide.model_validate_json(topics.load_topics()[tid]["study_guide"])


@pytest.mark.asyncio
async def test_incomplete_guide_not_cached(monkeypatch):
    replies = iter(['{"key_points": ["kp"], "terms": []}', "bad"])  # map ok, reduce fails

    async def complete(prompt, system=None, **kw):
        return next(replies)

    monkeypatch.setattr(llm, "complete", complete)
    tid = await _topic_with_text()
    res = await topics.generate_study_guide(tid)
    assert res["data"]["sections"] and res["data"]["overview"] == ""
    assert topics.load_topics()[tid]["study_guide"] is None


@pytest.mark.asyncio
async def test_all_units_failing_returns_could_not_generate(monkeypatch):
    async def complete(prompt, system=None, **kw):
        raise RuntimeError("llm down")

    monkeypatch.setattr(llm, "complete", complete)
    tid = await _topic_with_text()
    res = await topics.generate_study_guide(tid)
    assert "Could not generate" in res["guide"] and res["data"] is None
    assert topics.load_topics()[tid]["study_guide"] is None


@pytest.mark.asyncio
async def test_legacy_markdown_cache_regenerated(guide_llm):
    tid = await _topic_with_text()
    topics._update_topic(tid, study_guide="## Old markdown guide")
    res = await topics.generate_study_guide(tid)
    assert res["data"] is not None and guide_llm


@pytest.mark.asyncio
async def test_study_guide_without_chunks_is_not_cached(llm_calls):
    tid = topics.create_topic("Legacy")["id"]
    topics.add_source(tid, "note", "pre-Phase-1 source")  # a source row with no chunks

    res = await topics.generate_study_guide(tid)
    assert "Could not generate" in res["guide"] and res["data"] is None and llm_calls == []
    assert topics.load_topics()[tid]["study_guide"] is None


@pytest.mark.asyncio
async def test_suggest_topic_name_uses_llm():
    assert await topics.suggest_topic_name("Attention heads...", []) == "Transformer Models"


@pytest.mark.asyncio
async def test_grounded_output_has_no_passage_markers(monkeypatch):
    # Guides are cached without citations and scripts are read aloud: [n] would dangle
    systems = []

    async def complete(prompt, system=None, **kw):
        systems.append(system)
        return "Gradient descent steps downhill [1]. It uses a learning rate [2][3]."

    class OneHit:
        async def retrieve(self, q, topic_id, k=6):
            return [Hit(1, 1, "Notes", 2, None, "gradient descent", 1.0)]

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(query, "get_retriever", OneHit)
    tid = topics.create_topic("Optimisation")["id"]
    topics.add_source(tid, "note", "n")

    assert await topics.generate_audio_script(tid) == "Gradient descent steps downhill. It uses a learning rate."
    assert "square brackets" not in systems[0]
