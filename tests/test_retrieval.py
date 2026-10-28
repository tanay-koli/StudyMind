"""StudyMind — chunk store and retrievers (embedded Qdrant :memory:, fake models)."""

from qdrant_client import models
from sqlmodel import select

from core import retrieval, topics
from core.config import settings
from core.db import Chunk, session
from core.parse import Block


def _src(topic_id="general", title="Doc"):
    return topics.add_source(topic_id, topic_id, "note", title)["id"]


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
