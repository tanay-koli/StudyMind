"""Indexed chunks with planted concept triples, for tutor and agent tests."""

from core import concepts, retrieval, topics
from core.db import ConceptEdge, ConceptExtraction, session
from core.parse import Block


def plant(texts: list[str], edges: dict[int, list[tuple[str, str, str]]], topic: str = "Biology") -> tuple[str, list[int]]:
    """Index one chunk per text in a new topic (page = index + 1) and plant (head, relation, tail) edges.

    Returns (topic_id, chunk ids in text order).
    """
    tid = topics.create_topic(topic)["id"]
    sid = topics.add_source(tid, "note", f"{topic} notes")["id"]
    retrieval.index_chunks(sid, tid, [Block(i + 1, None, t) for i, t in enumerate(texts)])
    with session() as s:
        from sqlmodel import select
        from core.db import Chunk

        ids = list(s.exec(select(Chunk.id).where(Chunk.source_id == sid).order_by(Chunk.ord)).all())
        for i, triples in edges.items():
            for h, r, t in triples:
                s.add(ConceptEdge(chunk_id=ids[i], source_id=sid, topic_id=tid, head=h, relation=r, tail=t,
                                  head_norm=concepts.normalise(h), tail_norm=concepts.normalise(t)))
        for cid in ids:
            s.add(ConceptExtraction(chunk_id=cid))
        s.commit()
    return tid, ids
