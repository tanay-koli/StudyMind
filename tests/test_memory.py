"""StudyMind — learner facts: ADD / UPDATE / DELETE extraction (offline)."""

import pytest

from core import memory
from core.db import LearnerFact, session
from core.llm import llm


@pytest.fixture
def replies(monkeypatch):
    """Scripted complete_json replies (an Exception is raised)."""
    queue = []

    async def complete_json(prompt, system=None, **kw):
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete_json", complete_json)
    return queue


@pytest.mark.parametrize("message,expected", [
    ("My exam is on 12 Dec", True), ("I prefer short examples", True), ("i'm revising for finals", True),
    ("what is an enzyme", False), ("B", False), ("Explain the exam board's rules", True),
])
def test_should_remember(message, expected):
    assert memory.should_remember(message) is expected


@pytest.mark.asyncio
async def test_update_applies_ops(replies):
    replies += [{"ops": [{"op": "ADD", "fact": "Exam on 12 Dec"}, {"op": "ADD", "fact": "Prefers short examples"}]},
                {"ops": [{"op": "UPDATE", "id": 1, "fact": "Exam moved to 15 Dec"}, {"op": "DELETE", "id": 2}]}]
    assert await memory.update("my exam is on 12 Dec and I like short examples") == 2
    assert await memory.update("my exam moved; long examples are fine") == 2
    assert [f.fact for f in memory.facts()] == ["Exam moved to 15 Dec"]


@pytest.mark.asyncio
async def test_update_skips_invalid_ops(replies):
    replies += [{"ops": [{"op": "ADD", "fact": "Keep me"}]},
                {"ops": [{"op": "UPDATE", "id": 99, "fact": "x"}, {"op": "ADD", "fact": "  "}, {"op": "MERGE", "fact": "y"},
                         {"op": "DELETE"}]},
                ValueError("no json")]
    await memory.update("I keep notes")
    assert await memory.update("my notes") == 0
    assert await memory.update("my notes again") == 0
    assert [f.fact for f in memory.facts()] == ["Keep me"]


@pytest.mark.asyncio
async def test_fact_cap(replies):
    with session() as s:
        s.add_all(LearnerFact(fact=f"fact {i}") for i in range(memory.MAX_FACTS))
        s.commit()
    replies.append({"ops": [{"op": "ADD", "fact": "one too many"}]})
    assert await memory.update("I have another fact") == 0 and len(memory.facts()) == memory.MAX_FACTS


def test_profile_text_and_delete():
    assert memory.profile() == ""
    with session() as s:
        s.add_all([LearnerFact(fact="a"), LearnerFact(fact="b")])
        s.commit()
    assert memory.profile() == "About the student:\n- a\n- b"
    first = memory.facts()[0].id
    assert memory.delete(first) is True and memory.delete(first) is False
    assert memory.profile() == "About the student:\n- b"


@pytest.mark.asyncio
async def test_repeated_delete_does_not_open_the_cap_and_string_ids_work(replies):
    with session() as s:
        s.add_all(LearnerFact(fact=f"fact {i}") for i in range(memory.MAX_FACTS))
        s.commit()
    first = memory.facts()[0].id
    replies.append({"ops": [{"op": "DELETE", "id": first}, {"op": "DELETE", "id": first},
                            {"op": "ADD", "fact": "new one"}, {"op": "ADD", "fact": "one too many"},
                            {"op": "UPDATE", "id": str(memory.facts()[1].id), "fact": "updated by string id"}]})
    assert await memory.update("my facts changed") == 3
    texts = [f.fact for f in memory.facts()]
    assert len(texts) == memory.MAX_FACTS and "new one" in texts and "one too many" not in texts
    assert "updated by string id" in texts
