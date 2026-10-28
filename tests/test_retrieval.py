"""StudyMind — chunk store and retrievers (embedded Qdrant :memory:, fake models)."""

import pytest
from qdrant_client import models
from sqlmodel import select

from core import retrieval, topics
from core.config import settings
from core.db import Chunk, session
from core.parse import Block


def _src(topic_id="general", title="Doc"):
    return topics.add_source(topic_id, "note", title)["id"]


def _count(topic_id=None, collection=None):
    flt = None
    if topic_id:
        flt = models.Filter(must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))])
    return retrieval.store().count(collection or retrieval.collection_name(), count_filter=flt).count


def test_index_writes_rows_and_points():
    sid = _src()
    assert retrieval.index_chunks(sid, "general", [Block(1, "A", "alpha"), Block(2, "B", "beta")]) == 2
    assert _count() == 2
    with session() as s:
        rows = s.exec(select(Chunk).order_by(Chunk.ord)).all()
    assert [(r.source_id, r.page, r.section_heading, r.chunker, r.ord) for r in rows] == [
        (sid, 1, "A", "structure", 0), (sid, 2, "B", "structure", 1),
    ]


def test_sparse_query_respects_topic_filter():
    # Spike: embedded Qdrant handles IDF sparse vectors together with a payload filter
    t2 = topics.create_topic("Other")["id"]
    retrieval.index_chunks(_src(), "general", [Block(1, None, "self attention")])
    retrieval.index_chunks(_src(t2), t2, [Block(1, None, "self attention")])
    flt = models.Filter(must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=t2))])
    res = retrieval.store().query_points(
        retrieval.collection_name(), query=retrieval.embed_sparse(["attention"], query=True)[0],
        using="bm25", query_filter=flt, limit=10, with_payload=True,
    ).points
    assert [p.payload["topic_id"] for p in res] == [t2]


def test_delete_hits_every_collection(monkeypatch):
    sid = _src()
    retrieval.index_chunks(sid, "general", [Block(1, None, "alpha")])
    monkeypatch.setattr(settings, "embed_model", "nomic")
    assert retrieval.reindex() == 1
    retrieval.delete_source_points([sid])
    for name in ("chunks_bge_small", "chunks_nomic"):
        assert _count(collection=name) == 0


def test_move_updates_payload():
    t2 = topics.create_topic("Other")["id"]
    sid = _src()
    retrieval.index_chunks(sid, "general", [Block(1, None, "alpha")])
    retrieval.move_source_points(sid, t2)
    assert (_count("general"), _count(t2)) == (0, 1)


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def test_rrf():
    # 3: 1/63 + 1/61 > 2: 2/62 > 1: 1/61 > 4: 1/63
    assert retrieval.rrf([[1, 2, 3], [3, 2, 4]]) == [3, 2, 1, 4]


def _two_topics():
    t1 = topics.create_topic("T1")["id"]
    t2 = topics.create_topic("T2")["id"]
    retrieval.index_chunks(_src(t1), t1, [
        Block(4, "Attention", "self attention weighs tokens"),
        Block(5, "RNN", "recurrent hidden state"),
    ])
    retrieval.index_chunks(_src(t2, "Other"), t2, [Block(9, None, "attention in vision transformers")])
    return t1, t2


@pytest.mark.asyncio
async def test_hybrid_returns_relevant_chunk_with_page():
    t1, _ = _two_topics()
    hits = await retrieval.HybridRetriever().retrieve("what is self attention", t1, k=1)
    assert [(h.page, h.section, h.source_title) for h in hits] == [(4, "Attention", "Doc")]


@pytest.mark.asyncio
async def test_hybrid_topic_none_searches_all_topics():
    _two_topics()
    hits = await retrieval.HybridRetriever().retrieve("attention", None)
    assert {h.source_title for h in hits} == {"Doc", "Other"}


@pytest.mark.asyncio
async def test_hybrid_empty_store_returns_empty():
    assert await retrieval.HybridRetriever().retrieve("anything", "general") == []


@pytest.mark.asyncio
async def test_dense_retriever_returns_hits_without_rerank(monkeypatch):
    t1, _ = _two_topics()

    def no_rerank(query, texts):
        raise AssertionError("dense baseline must not rerank")

    monkeypatch.setattr(retrieval, "rerank", no_rerank)
    hits = await retrieval.DenseRetriever().retrieve("self attention weighs tokens", t1, k=1)
    assert [(h.page, h.section, h.source_title) for h in hits] == [(4, "Attention", "Doc")]


@pytest.mark.asyncio
async def test_dense_retriever_empty_store():
    assert await retrieval.DenseRetriever().retrieve("anything", None) == []


def test_get_retriever(monkeypatch):
    assert isinstance(retrieval.get_retriever(), retrieval.HybridRetriever)
    monkeypatch.setattr(settings, "retriever", "dense")
    assert isinstance(retrieval.get_retriever(), retrieval.DenseRetriever)


def test_concurrent_search_and_writes_do_not_fail():
    # Embedded Qdrant has no internal locking; ingest/delete and ask run in worker threads
    import threading

    retrieval.index_chunks(_src(), "general", [Block(1, None, f"seed text {i}") for i in range(20)])
    errors, stop = [], threading.Event()

    def writer():
        try:
            for i in range(30):
                sid = _src()
                retrieval.index_chunks(sid, "general", [Block(1, None, f"alpha beta {i} {j}") for j in range(5)])
                retrieval.delete_source_points([sid])
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                retrieval.HybridRetriever()._search("alpha beta", "general", 6)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_ensure_indexed_fills_gaps_after_model_switch(monkeypatch):
    retrieval.index_chunks(_src(), "general", [Block(1, None, "alpha")])
    monkeypatch.setattr(settings, "embed_model", "nomic")
    retrieval.ensure_indexed()  # new model: collection missing
    retrieval.index_chunks(_src(), "general", [Block(1, None, "beta")])
    monkeypatch.setattr(settings, "embed_model", "bge-small")
    retrieval.ensure_indexed()  # back to old model: missing the source added under nomic
    assert (_count(collection="chunks_bge_small"), _count(collection="chunks_nomic")) == (2, 2)


@pytest.mark.asyncio
async def test_heading_text_is_searchable():
    # Slide titles carry the topic; the body may never repeat it
    retrieval.index_chunks(_src(), "general", [
        Block(1, "Optimisers", "momentum and adaptive steps"),
        Block(2, "Backpropagation", "chain rule applied layer by layer"),
    ])
    hits = await retrieval.HybridRetriever().retrieve("backpropagation", "general", k=1)
    assert hits[0].section == "Backpropagation"
    assert hits[0].text == "chain rule applied layer by layer"  # citation text stays the body


def test_model_loaders_load_once_under_concurrency(monkeypatch):
    # First cross-topic call fans out one retrieval per topic; a cold cache must not load N copies
    import threading
    import time

    import fastembed

    loads = []

    class SlowModel:
        def __init__(self, *a, **kw):
            loads.append(a)
            time.sleep(0.05)

    monkeypatch.setattr(fastembed, "TextEmbedding", SlowModel)
    retrieval._load_dense.cache_clear()
    threads = [threading.Thread(target=retrieval._dense_model, args=("m",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    retrieval._load_dense.cache_clear()
    assert len(loads) == 1
