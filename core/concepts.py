"""
StudyMind — concept triple extraction.

Pending chunks go to the local LLM in batches; each yields (head, relation, tail)
triples stored in concept_edge, with a concept_extraction row marking it done.
"""

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass

from sqlmodel import select

from core.db import Chunk, ConceptEdge, ConceptExtraction, session
from core.llm import llm

log = logging.getLogger("studymind.concepts")

BATCH = 4
MAX_TRIPLES = 5
MAX_PART = 60
CONNECT_K = 15
TOP_SHARED = 8
PENDING_NOTE = "{n} of {m} passages are still being processed — try again shortly."


def normalise(name: str) -> str:
    # shortcut: no lemmatisation or synonyms, upgrade if topic maps look fragmented
    words = re.sub(r"[^\w\s]", "", name.lower()).split()
    if len(words) > 1 and words[0] in ("the", "a", "an"):
        words = words[1:]
    words = [w[:-1] if w.endswith("s") and not w.endswith(("ss", "is", "us")) else w for w in words]
    return " ".join(words)


def _prompt(texts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    return (
        f"{numbered}\n\n"
        f"For each numbered passage, extract at most {MAX_TRIPLES} (concept, relation, concept) triples. "
        "Concepts are short noun phrases. Reply with JSON only: "
        '{"chunks": [{"n": 1, "triples": [["head", "relation", "tail"]]}]}'
    )


def _valid(t) -> bool:
    return (isinstance(t, (list, tuple)) and len(t) == 3
            and all(isinstance(p, str) and p.strip() and len(p.strip()) <= MAX_PART for p in t)
            and all(normalise(p) for p in t) and normalise(t[0]) != normalise(t[2]))


async def extract_pending() -> int:
    """Extract triples for every pending chunk; return how many chunks were marked done."""
    failed: set[int] = set()
    done = 0
    while True:
        with session() as s:
            q = select(Chunk).where(~Chunk.id.in_(select(ConceptExtraction.chunk_id)))
            if failed:
                q = q.where(~Chunk.id.in_(failed))
            batch = s.exec(q.order_by(Chunk.id).limit(BATCH)).all()
        if not batch:
            return done
        try:
            reply = await llm.complete_json(_prompt([c.text for c in batch]), timeout=180)
            by_n = {}
            for r in reply["chunks"]:
                with contextlib.suppress(TypeError, ValueError):
                    by_n[int(r["n"])] = r.get("triples")
        except Exception as e:
            # shortcut: a chunk that always fails is retried on every wake/restart, add an attempt count if logs show repeats
            log.warning("concept batch %s failed: %s", [c.id for c in batch], e)
            failed.update(c.id for c in batch)
            continue
        try:
            with session() as s:
                # ids are reused rowids: only write chunks whose text is still what we sent
                now = dict(s.exec(select(Chunk.id, Chunk.text).where(Chunk.id.in_([c.id for c in batch]))).all())
                for n, c in enumerate(batch, 1):
                    if now.get(c.id) != c.text:
                        continue
                    if n not in by_n:
                        failed.add(c.id)  # model omitted it: stay pending, retry next pass
                        continue
                    triples = by_n[n]
                    triples = [[p.strip() for p in t] for t in triples if _valid(t)] if isinstance(triples, list) else []
                    for h, r, t in triples[:MAX_TRIPLES]:
                        s.add(ConceptEdge(chunk_id=c.id, source_id=c.source_id, topic_id=c.topic_id, head=h,
                                          relation=r, tail=t, head_norm=normalise(h), tail_norm=normalise(t)))
                    s.add(ConceptExtraction(chunk_id=c.id))
                    done += 1
                s.commit()
        except Exception:
            log.warning("concept write for batch %s failed", [c.id for c in batch], exc_info=True)
            failed.update(c.id for c in batch)


@dataclass
class Shared:
    name: str
    support: int
    edge_a: ConceptEdge
    edge_b: ConceptEdge


def shared_concepts(chunks_a: list[int], chunks_b: list[int], exclude: set[str]) -> list[Shared]:
    """Concepts mentioned (as head or tail) on both sides' chunks, most-supported first."""
    a, b = set(chunks_a), set(chunks_b)
    with session() as s:
        edges = s.exec(select(ConceptEdge).where(ConceptEdge.chunk_id.in_(a | b)).order_by(ConceptEdge.id)).all()
    found: dict[str, dict] = {}
    for e in edges:
        for norm, shown in ((e.head_norm, e.head), (e.tail_norm, e.tail)):
            if norm in exclude:
                continue
            f = found.setdefault(norm, {"chunks": set(), "A": {}, "B": {}, "name": None})
            f["chunks"].add(e.chunk_id)
            for side, ids in (("A", a), ("B", b)):
                if e.chunk_id in ids:
                    f[side].setdefault(e.chunk_id, e)
                    if side == "A" and f["name"] is None:
                        f["name"] = shown
    out = []
    for f in found.values():
        # shared needs an A chunk and a different B chunk, else one overlap chunk counts on both sides
        pair = next(((ea, eb) for ca, ea in f["A"].items() for cb, eb in f["B"].items() if ca != cb), None)
        if pair:
            out.append((f, pair))
    out.sort(key=lambda x: (-len(x[0]["chunks"]), x[0]["name"]))
    return [Shared(f["name"], len(f["chunks"]), ea, eb) for f, (ea, eb) in out[:TOP_SHARED]]


def pending_count(chunk_ids: list[int]) -> int:
    with session() as s:
        done = set(s.exec(select(ConceptExtraction.chunk_id).where(ConceptExtraction.chunk_id.in_(chunk_ids))).all())
    return len(set(chunk_ids) - done)


_task: asyncio.Task | None = None
_event: asyncio.Event | None = None


async def _run(event: asyncio.Event) -> None:
    while True:
        try:
            await extract_pending()
        except Exception:
            log.exception("concept extraction pass failed")
        await event.wait()
        event.clear()


def start() -> asyncio.Task:
    global _task, _event
    _event = asyncio.Event()
    _task = asyncio.create_task(_run(_event))
    return _task


def wake() -> None:
    if _event is not None:
        _event.set()


async def stop() -> None:
    global _task, _event
    task, _task, _event = _task, None, None
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
