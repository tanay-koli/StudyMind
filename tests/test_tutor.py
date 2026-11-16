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
    assert await tutor.pick_targets(tid, review=True, today=TODAY) == [
        ("delta", ids[3]), ("alpha", ids[0]), ("beta", ids[1])]


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


# ---------------------------------------------------------------------------
# Questions and grading
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_script(monkeypatch):
    """Scripted llm.complete / complete_json replies (an Exception is raised); records (prompt, system)."""
    from core.llm import llm

    s = {"text": [], "json": [], "calls": []}

    async def complete(prompt, system=None, **kw):
        s["calls"].append((prompt, system))
        return s["text"].pop(0)

    async def complete_json(prompt, system=None, **kw):
        s["calls"].append((prompt, system))
        reply = s["json"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "complete_json", complete_json)
    return s


MCQ = {"question": "What do enzymes lower?", "options": ["Activation energy", "Temperature", "pH", "Mass"],
       "answer": "A", "difficulty": "easy"}


@pytest.mark.asyncio
async def test_generate_valid_mcq_and_short(llm_script):
    from tests.planting import plant

    _, ids = plant(["Enzymes lower activation energy."], {0: [("Enzyme", "lowers", "Activation energy")]})
    llm_script["json"] = [dict(MCQ, options=["A) Activation energy", "B) Temperature", "C) pH", "D) Mass"]),
                          {"question": "Why are enzymes useful?", "answer": "They lower activation energy.",
                           "difficulty": "medium"}]
    q = await tutor.generate_question("enzyme", ids[0], "mcq", facts="About the student:\n- likes short answers")
    assert (q.kind, q.options[0], q.answer, q.concept, q.chunk_id) == (
        "mcq", "Activation energy", "A", "enzyme", ids[0])
    assert q.citation["chunk_id"] == ids[0] and "lower activation energy" in q.citation["text"]
    assert "likes short answers" in llm_script["calls"][0][1]
    short = await tutor.generate_question("enzyme", ids[0], "short")
    assert short.kind == "short" and short.options == [] and short.difficulty == "medium"


@pytest.mark.asyncio
async def test_generate_rejects_bad_mcq_twice(llm_script):
    from tests.planting import plant

    _, ids = plant(["Enzymes lower activation energy."], {0: [("Enzyme", "lowers", "Activation energy")]})
    llm_script["json"] = [dict(MCQ, options=["x", "y", "z"]), ValueError("no json")]
    assert await tutor.generate_question("enzyme", ids[0], "mcq") is None and len(llm_script["calls"]) == 2


@pytest.mark.parametrize("answer,letter", [
    ("b", "B"), ("(B)", "B"), ("B.", "B"), ("I think it's B because a cell divides", "B"),
    ("pH", "C"), ("  temperature ", "B"), ("no idea", None),
])
def test_mcq_letter_forms(answer, letter):
    assert tutor._mcq_letter(answer, MCQ["options"]) == letter


@pytest.mark.asyncio
async def test_grade_mcq_correct_makes_no_call(llm_script):
    q = _question(kind="mcq", options=MCQ["options"], answer="A")
    assert await tutor.grade(q, "a") == ("correct", "") and llm_script["calls"] == []


@pytest.mark.asyncio
async def test_grade_mcq_wrong_explains_once(llm_script):
    llm_script["text"] = ["  Enzymes lower activation energy, as the passage says.  "]
    q = _question(kind="mcq", options=MCQ["options"], answer="A")
    assert await tutor.grade(q, "C") == ("wrong", "Enzymes lower activation energy, as the passage says.")
    assert len(llm_script["calls"]) == 1 and "Activation energy" in llm_script["calls"][0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["correct", "partial", "wrong"])
async def test_grade_short_verdicts(llm_script, verdict):
    llm_script["json"] = [{"verdict": verdict, "feedback": "One line."}]
    assert await tutor.grade(_question(), "they help") == (verdict, "One line.")


@pytest.mark.asyncio
async def test_grade_short_invalid_twice_is_ungraded(llm_script):
    llm_script["json"] = [{"verdict": "maybe"}, ValueError("no json")]
    assert await tutor.grade(_question(), "they help") == (None, "Couldn't grade this answer.")


@pytest.mark.asyncio
async def test_grade_uses_stored_citation_text(llm_script):
    # The chunk is gone (source deleted mid-quiz); grading reads the question's own citation
    llm_script["json"] = [{"verdict": "correct", "feedback": "Yes."}]
    await tutor.grade(_question(chunk_id=999), "lower activation energy")
    assert "Enzymes lower activation energy." in llm_script["calls"][0][0]


@pytest.mark.parametrize("answer,options,letter", [
    ("d) zinc", ["iron", "copper", "lead", "zinc"], "D"),
    ("(c) lead", ["iron", "copper", "lead", "zinc"], "C"),
    ("Vitamin C", ["Vitamin C", "Iron", "Zinc", "Calcium"], "A"),   # option text before a stray capital letter
])
def test_mcq_letter_more_forms(answer, options, letter):
    assert tutor._mcq_letter(answer, options) == letter


@pytest.mark.asyncio
async def test_connection_errors_are_not_treated_as_bad_replies(llm_script):
    # Ollama down mid-quiz must surface as an error (the agent keeps the quiz), not as "ungraded" or "no question"
    llm_script["json"] = [ConnectionError("ollama down")]
    with pytest.raises(ConnectionError):
        await tutor.grade(_question(), "they help")
    from tests.planting import plant

    _, ids = plant(["Enzymes lower activation energy."], {0: [("Enzyme", "lowers", "Activation energy")]})
    llm_script["json"] = [TimeoutError("timed out")]
    with pytest.raises(TimeoutError):
        await tutor.generate_question("enzyme", ids[0], "mcq")


@pytest.mark.asyncio
async def test_quiz_without_subject_uses_the_topics_own_concepts():
    # Passages rarely contain their topic's name, so "quiz me" must not search for it
    from tests.planting import plant

    tid, ids = plant(["Enzymes bind substrates.", "Glycolysis splits glucose."],
                     {0: [("Enzyme", "binds", "Substrate")], 1: [("Glycolysis", "splits", "Glucose")]}, topic="Biology")
    _mastery("enzyme", "correct", 2.8, TODAY + timedelta(6))  # already known well: asked last
    targets = await tutor.pick_targets(tid, n=5)
    assert [c for c, _ in targets] == ["substrate", "glycolysis", "glucose", "enzyme"]
    assert dict(targets)["glycolysis"] == ids[1] and dict(targets)["enzyme"] == ids[0]
    other, _ = plant(["Cells divide."], {0: [("Cell", "undergoes", "Mitosis")]}, topic="Other")
    assert {c for c, _ in await tutor.pick_targets(None, n=10)} == {"enzyme", "substrate", "glycolysis", "glucose",
                                                                   "cell", "mitosis"}


@pytest.mark.parametrize("answer,letter", [
    ("A cell divides into two", None),          # the article "A", not option A
    ("I think it's A because cells divide", "A"),
])
def test_mcq_letter_article_a(answer, letter):
    assert tutor._mcq_letter(answer, ["w", "x", "y", "z"]) == letter
