"""StudyMind — tutor: SM-2 scheduling, attempts, targets, questions and grading (offline)."""

from datetime import date, timedelta

import pytest
from sqlmodel import select

from core import tutor
from core.db import ConceptMastery, QuizAttempt, session

TODAY = date(2026, 11, 3)


def _question(concept="enzyme", kind="short", chunk_id=1, **kw):
    fields = dict(kind=kind, question="What do enzymes do?", options=[], answer="Lower activation energy",
                  difficulty="easy", concept=concept, chunk_id=chunk_id,
                  citation={"source_title": "Notes", "page": 1, "chunk_id": chunk_id,
                            "text": "Enzymes lower activation energy.", "kind": "text"})
    fields.update(kw)
    return tutor.Question(**fields)


def test_sm2_correct_three_times():
    row = ConceptMastery(concept="x", next_review=TODAY.isoformat())
    seen = []
    for _ in range(3):
        tutor.sm2(row, "correct", TODAY)
        seen.append((row.interval_days, round(row.ease, 2), row.next_review))
    assert seen == [(1, 2.6, (TODAY + timedelta(1)).isoformat()), (6, 2.7, (TODAY + timedelta(6)).isoformat()),
                    (16, 2.8, (TODAY + timedelta(16)).isoformat())]
    assert row.repetitions == 3 and row.attempts == 3 and row.wrong == 0


def test_sm2_partial_and_wrong():
    partial = tutor.sm2(ConceptMastery(concept="p", next_review=TODAY.isoformat()), "partial", TODAY)
    assert partial.interval_days == 1 and partial.ease == pytest.approx(2.36) and partial.wrong == 1
    wrong = tutor.sm2(ConceptMastery(concept="w", next_review=TODAY.isoformat()), "wrong", TODAY)
    assert wrong.interval_days == 1 and wrong.ease == pytest.approx(1.96) and wrong.repetitions == 0
    for _ in range(4):
        tutor.sm2(wrong, "wrong", TODAY)
    assert wrong.ease == 1.3 and wrong.last_verdict == "wrong" and wrong.wrong == 5


def test_record_writes_attempt_and_mastery():
    q = _question()
    tutor.record(q, "it speeds reactions", "correct", "", "general", today=TODAY)
    tutor.record(q, "no idea", "wrong", "It lowers activation energy.", "general", today=TODAY)
    tutor.record(q, "???", None, "Couldn't grade this answer.", "general", today=TODAY)
    with session() as s:
        attempts = s.exec(select(QuizAttempt).order_by(QuizAttempt.id)).all()
        m = s.get(ConceptMastery, "enzyme")
    assert [(a.verdict, a.answer, a.kind, a.chunk_id) for a in attempts] == [
        ("correct", "it speeds reactions", "short", 1), ("wrong", "no idea", "short", 1), (None, "???", "short", 1)]
    assert (m.attempts, m.wrong, m.last_verdict, m.interval_days) == (2, 1, "wrong", 1)
