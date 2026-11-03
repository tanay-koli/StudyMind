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


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _mastery(concept, verdict, ease, next_review, wrong=0):
    with session() as s:
        s.add(ConceptMastery(concept=concept, last_verdict=verdict, ease=ease, next_review=next_review.isoformat(),
                             attempts=1, wrong=wrong))
        s.commit()


@pytest.mark.asyncio
async def test_quiz_targets_from_retrieved_chunks():
    from core import concepts
    from tests.planting import plant

    tid, ids = plant(
        ["Enzymes bind substrates at the active site.", "Enzyme activity falls when an inhibitor binds.",
         "Glucose is broken down in glycolysis."],
        {0: [("Enzyme", "binds", "Substrate")], 1: [("Enzyme", "blocked by", "Inhibitor")],
         2: [("Glucose", "broken down in", "Glycolysis")]})
    targets = await tutor.pick_targets(tid, about="enzyme", n=5)
    names = [c for c, _ in targets]
    assert len(names) == len(set(names)) <= 5 and names.count("enzyme") == 1  # distinct by concept
    assert all(cid in concepts.chunks_for_concept(c) for c, cid in targets)
    assert names[0] == "enzyme"  # the best-ranked chunk mentions "enzyme"


@pytest.mark.asyncio
async def test_review_targets_only_weak_or_due_in_order():
    from tests.planting import plant

    tid, ids = plant(["a", "b", "c", "d"], {0: [("Alpha", "r", "x0")], 1: [("Beta", "r", "x1")],
                                            2: [("Gamma", "r", "x2")], 3: [("Delta", "r", "x3")]})
    tomorrow = TODAY + timedelta(1)
    _mastery("alpha", "wrong", 1.96, tomorrow, wrong=1)
    _mastery("beta", "partial", 2.36, tomorrow, wrong=1)
    _mastery("gamma", "correct", 2.6, tomorrow)
    _mastery("delta", "correct", 2.6, TODAY)  # due today
    assert await tutor.pick_targets(tid, review=True, today=TODAY) == [("delta", ids[3]), ("alpha", ids[0]), ("beta", ids[1])]


@pytest.mark.asyncio
async def test_review_respects_topic():
    from tests.planting import plant

    plant(["a"], {0: [("Alpha", "r", "x0")]}, topic="Topic X")
    other, _ = plant(["b"], {0: [("Beta", "r", "x1")]}, topic="Topic Y")
    _mastery("alpha", "wrong", 1.96, TODAY + timedelta(1), wrong=1)
    assert await tutor.pick_targets(other, review=True, today=TODAY) == []
    assert [c for c, _ in await tutor.pick_targets(None, review=True, today=TODAY)] == ["alpha"]


@pytest.mark.asyncio
async def test_review_empty_when_nothing_due():
    from tests.planting import plant

    tid, _ = plant(["a"], {0: [("Alpha", "r", "x0")]})
    _mastery("alpha", "correct", 2.6, TODAY + timedelta(6))
    assert await tutor.pick_targets(tid, review=True, today=TODAY) == []
