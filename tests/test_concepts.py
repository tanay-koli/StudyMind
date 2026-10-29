"""StudyMind — concept triple extraction."""

import asyncio
import json
import math

import pytest
from sqlmodel import select

from core import concepts, topics
from core.db import Chunk, ConceptEdge, ConceptExtraction, Source, session
from core.llm import llm


@pytest.mark.parametrize("raw,want", [
    ("The Neural Networks", "neural network"),
    ("TCP/IP", "tcpip"),
    ("  Gradient   Descent ", "gradient descent"),
    ("analysis", "analysis"),
    ("class", "class"),
    ("virus", "virus"),
    ("an Enzyme", "enzyme"),
])
def test_normalise(raw, want):
    assert concepts.normalise(raw) == want


def _make_chunks(n, topic="Concept Topic"):
    tid = topics.create_topic(topic)["id"]
    with session() as s:
        src = Source(topic_id=tid, source_type="note", title="t")
        s.add(src)
        s.commit()
        chunks = [Chunk(source_id=src.id, topic_id=tid, chunker="x", ord=i, text=f"text {i}") for i in range(n)]
        s.add_all(chunks)
        s.commit()
        return tid, src.id, [c.id for c in chunks]


@pytest.mark.asyncio
async def test_extract_stores_edges_and_done_rows(monkeypatch):
    tid, sid, cids = _make_chunks(5)
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"chunks": [
                {"n": 1, "triples": [
                    ["The Neural Networks", "use", "Gradient Descent"],
                    ["A", "x", "a"],            # head == tail
                    ["", "x", "B"],             # empty part
                    ["C", "x", "D" * 61],       # too long
                    ["E", "x", "F"], ["G", "x", "H"], ["I", "x", "J"], ["K", "x", "L"],
                    ["M", "x", "N"],            # 6th valid triple
                ]},
                {"n": 2, "triples": []}, {"n": 3, "triples": []}, {"n": 4, "triples": []},
            ]})
        return json.dumps({"chunks": [{"n": 1, "triples": [["P", "r", "Q"]]}]})

    monkeypatch.setattr(llm, "complete", complete)
    assert await concepts.extract_pending() == 5
    assert len(calls) == 2
    with session() as s:
        edges = s.exec(select(ConceptEdge)).all()
        done = {e.chunk_id for e in s.exec(select(ConceptExtraction)).all()}
    assert done == set(cids)
    first = [e for e in edges if e.chunk_id == cids[0]]
    assert len(first) == 5 and first[0].head_norm == "neural network" and first[0].tail_norm == "gradient descent"
    assert all(e.source_id == sid and e.topic_id == tid for e in edges)
    assert [e.head for e in edges if e.chunk_id == cids[4]] == ["P"]


@pytest.mark.asyncio
async def test_failed_batch_stays_pending_and_pass_ends(monkeypatch):
    _make_chunks(9)
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        return "not json"

    monkeypatch.setattr(llm, "complete", complete)
    assert await concepts.extract_pending() == 0
    assert len(calls) == math.ceil(9 / 4)
    with session() as s:
        assert s.exec(select(ConceptExtraction)).first() is None


@pytest.mark.asyncio
async def test_deleted_chunk_skipped(monkeypatch):
    _, sid, _ = _make_chunks(2)

    async def complete(prompt, system=None, **kw):
        topics.delete_source(sid)
        return json.dumps({"chunks": [{"n": 1, "triples": [["A", "r", "B"]]}]})

    monkeypatch.setattr(llm, "complete", complete)
    await concepts.extract_pending()
    with session() as s:
        assert s.exec(select(ConceptEdge)).first() is None
        assert s.exec(select(ConceptExtraction)).first() is None


@pytest.mark.asyncio
async def test_reused_id_with_new_text_not_written(monkeypatch):
    _, sid, cids = _make_chunks(1)

    async def complete(prompt, system=None, **kw):
        topics.delete_source(sid)
        with session() as s:
            s.add(Chunk(id=cids[0], source_id=99, topic_id="x", chunker="x", ord=0, text="different"))
            s.commit()
        return json.dumps({"chunks": [{"n": 1, "triples": [["A", "r", "B"]]}]})

    monkeypatch.setattr(llm, "complete", complete)
    assert await concepts.extract_pending() == 0
    with session() as s:
        assert s.exec(select(ConceptEdge)).first() is None
        assert s.exec(select(ConceptExtraction)).first() is None


@pytest.mark.asyncio
async def test_string_n_ok_and_omitted_chunk_stays_pending(monkeypatch):
    _, _, cids = _make_chunks(2)
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        return json.dumps({"chunks": [{"n": "1", "triples": [["A", "r", "B"]]}, {"n": "x", "triples": []}]})

    monkeypatch.setattr(llm, "complete", complete)
    assert await concepts.extract_pending() == 1
    assert len(calls) == 1  # omitted chunk not retried in the same pass
    with session() as s:
        assert [e.chunk_id for e in s.exec(select(ConceptExtraction)).all()] == [cids[0]]


@pytest.mark.asyncio
async def test_unnormalisable_part_rejected(monkeypatch):
    _make_chunks(1)

    async def complete(prompt, system=None, **kw):
        return json.dumps({"chunks": [{"n": 1, "triples": [["%", "r", "B"], ["A", "r", "B"]]}]})

    monkeypatch.setattr(llm, "complete", complete)
    await concepts.extract_pending()
    with session() as s:
        assert [e.head for e in s.exec(select(ConceptEdge)).all()] == ["A"]


@pytest.mark.asyncio
async def test_write_failure_does_not_escape(monkeypatch):
    _make_chunks(1)

    async def complete(prompt, system=None, **kw):
        return json.dumps({"chunks": [{"n": 1, "triples": [["A", "r", "B"]]}]})

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(concepts, "normalise", boom)
    assert await concepts.extract_pending() == 0


@pytest.mark.asyncio
async def test_worker_wakes_on_ingest(monkeypatch):
    async def fake(prompt, system=None, **kw):
        return json.dumps({"chunks": [{"n": 1, "triples": [["a", "rel", "b"]]}]})
    monkeypatch.setattr(llm, "complete", fake)
    from core import ingest
    concepts.start()
    try:
        r = await ingest.ingest_text("Transformers use attention over tokens.", "n")
        assert r["status"] == "success"
        for _ in range(500):
            with session() as s:
                if s.exec(select(ConceptExtraction)).first():
                    break
            await asyncio.sleep(0.01)
        with session() as s:
            assert s.exec(select(ConceptExtraction)).first()
    finally:
        await concepts.stop()


def test_wake_without_start_is_noop():
    concepts.wake()


@pytest.mark.asyncio
async def test_stop_cancels_cleanly():
    concepts.start()
    await asyncio.sleep(0.05)
    await concepts.stop()
    await concepts.stop()
