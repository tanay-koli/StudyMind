"""
StudyMind — Tutor

Quiz questions from the learner's material, graded answers, and SM-2 spaced
repetition per concept (concepts come from the Phase 2 concept triples).
"""

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel

from core.db import ConceptMastery, QuizAttempt, session

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
