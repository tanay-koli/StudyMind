"""
StudyMind — Tutor

Quiz questions from the learner's material, graded answers, and SM-2 spaced
repetition per concept (concepts come from the Phase 2 concept triples).
"""

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel
from sqlmodel import select

from core import concepts, query
from core.db import ConceptMastery, QuizAttempt, session
from core.retrieval import get_retriever, relevant

SESSION_SIZE = 5
QUALITY = {"correct": 5, "partial": 3, "wrong": 1}
MIN_EASE = 1.3


class Question(BaseModel):
    kind: Literal["mcq", "short"]
    question: str
    options: list[str] = []  # 4 for mcq, shown as A–D
    answer: str  # mcq: the right option's letter; short: the expected answer
    difficulty: Literal["easy", "medium", "hard"]
    concept: str
    chunk_id: int
    citation: dict


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

async def pick_targets(topic_id: Optional[str], about: Optional[str] = None, review: bool = False,
                       n: int = SESSION_SIZE, today: Optional[date] = None) -> list[tuple[str, int]]:
    """(concept, chunk_id) pairs to quiz on.

    Quiz: concepts of the passages retrieved for `about` (else the topic name), in rank order.
    Review: concepts that are due or were last answered wrong / partly, weakest first; [] if none.
    """
    if review:
        today_s = (today or date.today()).isoformat()
        with session() as s:
            rows = s.exec(select(ConceptMastery).where(
                (ConceptMastery.next_review <= today_s) | ConceptMastery.last_verdict.in_(["wrong", "partial"]))).all()
        targets = []
        for row in sorted(rows, key=lambda r: (r.next_review, r.ease, -r.wrong, r.concept)):
            if chunks := concepts.chunks_for_concept(row.concept, topic_id):
                targets.append((row.concept, chunks[0]))
        return targets[:n]

    q = about or (query._topic_name(topic_id) if topic_id else "overview")
    hits = relevant(await get_retriever(rewrite=False).retrieve(q, topic_id, k=12))
    tags = concepts.chunk_concepts([h.chunk_id for h in hits])
    targets, seen = [], set()
    for h in hits:
        for c in tags.get(h.chunk_id, []):
            if c not in seen:
                seen.add(c)
                targets.append((c, h.chunk_id))
    return targets[:n]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def sm2(row: ConceptMastery, verdict: str, today: date) -> ConceptMastery:
    """SM-2 update for one graded answer (correct 5, partial 3, wrong 1)."""
    q = QUALITY[verdict]
    if q < 3:
        row.repetitions, row.interval_days = 0, 1
    else:
        row.repetitions += 1
        # the interval uses the ease from before this answer
        row.interval_days = {1: 1, 2: 6}.get(row.repetitions) or round(row.interval_days * row.ease)
    row.ease = max(MIN_EASE, row.ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    row.next_review = (today + timedelta(days=row.interval_days)).isoformat()
    row.last_verdict = verdict
    row.attempts += 1
    row.wrong += verdict != "correct"
    return row


def record(question: Question, answer: str, verdict: Optional[str], feedback: str,
           topic_id: Optional[str], today: Optional[date] = None) -> None:
    """Log the attempt; a graded one also moves the concept's SM-2 schedule."""
    today = today or date.today()
    with session() as s:
        s.add(QuizAttempt(topic_id=topic_id, concept=question.concept, chunk_id=question.chunk_id,
                          kind=question.kind, question=question.question, answer=answer,
                          verdict=verdict, feedback=feedback))
        if verdict is not None:
            row = s.get(ConceptMastery, question.concept) or ConceptMastery(
                concept=question.concept, next_review=today.isoformat())
            s.add(sm2(row, verdict, today))
        s.commit()
