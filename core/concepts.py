"""
StudyMind — concept triple extraction.

Pending chunks go to the local LLM in batches; each yields (head, relation, tail)
triples stored in concept_edge, with a concept_extraction row marking it done.
"""

import logging
import re

from sqlmodel import select

from core.db import Chunk, ConceptEdge, ConceptExtraction, session
from core.llm import llm

log = logging.getLogger("studymind.concepts")

BATCH = 4
MAX_TRIPLES = 5
MAX_PART = 60


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
            and normalise(t[0]) != normalise(t[2]))


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
            by_n = {r.get("n"): r.get("triples") for r in reply["chunks"]}
        except Exception as e:
            # shortcut: a chunk that always fails is retried on every wake/restart, add an attempt count if logs show repeats
            log.warning("concept batch %s failed: %s", [c.id for c in batch], e)
            failed.update(c.id for c in batch)
            continue
        with session() as s:
            alive = set(s.exec(select(Chunk.id).where(Chunk.id.in_([c.id for c in batch]))).all())
            for n, c in enumerate(batch, 1):
                if c.id not in alive:
                    continue
                triples = by_n.get(n)
                triples = [[p.strip() for p in t] for t in triples if _valid(t)] if isinstance(triples, list) else []
                for h, r, t in triples[:MAX_TRIPLES]:
                    s.add(ConceptEdge(chunk_id=c.id, source_id=c.source_id, topic_id=c.topic_id, head=h,
                                      relation=r, tail=t, head_norm=normalise(h), tail_norm=normalise(t)))
                s.add(ConceptExtraction(chunk_id=c.id))
                done += 1
            s.commit()
