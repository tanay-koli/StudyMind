"""StudyMind — the LangGraph agent: routing, corrective retrieval, self-check, persistence (offline)."""

import asyncio
import re

import pytest

from core import agent, topics
from core.llm import llm


@pytest.mark.parametrize("msg,topics,hint,expected", [
    ("quiz me on attention", None, None, ("quiz", [])),
    ("Can you test me?", None, None, ("quiz", [])),
    ("Review my weak topics", None, None, ("review", [])),
    ("Explain the peer review process", None, None, ("explain", [])),     # "review" inside a question
    ("What does the quiz in ch. 3 cover", None, None, ("qa", [])),          # "quiz" inside a question
    ("What is the difference between TCP and UDP?", None, None, ("explain", [])),  # "between" is not connect
    ("what links these?", ["Biology", "Chemistry"], None, ("connect", ["Biology", "Chemistry"])),
    ("What is the connection between attention and memory?", None, None, ("connect", ["attention", "memory"])),
    ("connect enzymes with catalysts", None, None, ("connect", ["enzymes", "catalysts"])),
    ("Explain backprop", None, None, ("explain", [])),
    ("how does dropout work", None, None, ("explain", [])),
    ("list the layers in the encoder", None, None, ("qa", [])),
    ("Explain backprop", None, "qa", ("qa", [])),            # UI hint wins
    ("what links these?", ["Biology"], None, ("qa", [])),     # one topic is not a connect
])
def test_classify(msg, topics, hint, expected):
    assert agent.classify(msg, topics, hint) == expected


# ---------------------------------------------------------------------------
# Graph (scripted search results and LLM replies)
# ---------------------------------------------------------------------------

def _doc(score, text="Attention weighs tokens.", cid=1):
    return {"chunk_id": cid, "source_id": 1, "source_title": "Notes", "page": 2, "section": None,
            "text": text, "score": score}


@pytest.fixture
def script(monkeypatch, tmp_path):
    """Scripted search results and LLM replies; returns (script, call log, runner)."""
    log = {"queries": [], "prompts": [], "checks": []}
    s = {"search": [], "complete": [], "json": []}

    async def search(q, topic_id):
        log["queries"].append(q)
        return s["search"].pop(0) if s["search"] else []

    async def complete(prompt, system=None, **kw):
        log["prompts"].append((prompt, system))
        return s["complete"].pop(0)

    async def complete_json(prompt, system=None, **kw):
        log["checks"].append(prompt)
        return s["json"].pop(0) if s["json"] else {"unsupported": []}

    monkeypatch.setattr(agent, "search_material", search)
    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "complete_json", complete_json)
    monkeypatch.setattr(agent.settings, "retriever", "hybrid")

    async def run(msg, thread="t1", **kw):
        async with agent.open_graph(str(tmp_path / "agent.db")) as g:
            cfg = {"configurable": {"thread_id": thread}}
            nodes = [n async for ev in g.astream({"messages": [], "question": msg, "topics": [], **kw}, cfg,
                                                  stream_mode="updates") for n in ev]
            return nodes, (await g.aget_state(cfg)).values
    return s, log, run


@pytest.mark.asyncio
async def test_strong_retrieval_answers_without_rewrite(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["Attention weighs tokens [1]."]
    nodes, st = await run("what does attention do")
    assert nodes == ["route_intent", "retrieve", "grade_documents", "generate", "self_check", "respond", "remember"]
    assert st["answer"] == "Attention weighs tokens [1]." and st["warning"] is None
    assert st["citations"][0]["chunk_id"] == 1
    assert st["messages"][-1] == {"role": "assistant", "content": "Attention weighs tokens [1]."}


@pytest.mark.asyncio
async def test_weak_then_strong_rewrites_once(script):  # done-when 1, offline
    s, log, run = script
    s["search"] = [[_doc(-6.0)], [_doc(1.5)]]
    s["complete"] = ["self-attention mechanism transformer", "Answer [1]."]
    nodes, st = await run("that thing from lecture")
    assert nodes.count("retrieve") == 2 and nodes.count("rewrite_query") == 1
    assert log["queries"] == ["that thing from lecture", "self-attention mechanism transformer"]


@pytest.mark.asyncio
async def test_always_weak_gives_up_after_two_rewrites(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)]] * 3
    s["complete"] = ["q2", "q3"]
    nodes, st = await run("vague")
    assert nodes.count("rewrite_query") == 2 and "not_found" in nodes and "generate" not in nodes
    assert st["answer"].startswith("This isn't in your material.")
    assert all(q in st["answer"] for q in ["vague", "q2", "q3"])


@pytest.mark.asyncio
async def test_zero_hits_skips_rewrite(script):
    s, log, run = script
    nodes, st = await run("anything")
    assert "rewrite_query" not in nodes and log["prompts"] == []
    assert st["answer"].startswith("This isn't in your material.")


@pytest.mark.asyncio
async def test_empty_rewrite_stops_instead_of_repeating_the_search(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)], [_doc(2.0)]]
    s["complete"] = ["   "]
    nodes, st = await run("vague")
    assert log["queries"] == ["vague"] and "not_found" in nodes


@pytest.mark.asyncio
async def test_self_check_catches_planted_claim(script):  # done-when 2
    s, log, run = script
    planted = "Attention was invented in 1850 [1]."
    s["search"] = [[_doc(2.0)]]
    s["complete"] = [f"Attention weighs tokens [1]. {planted}", f"Attention weighs tokens [1]. {planted}"]
    s["json"] = [{"unsupported": [planted]}, {"unsupported": [planted]}]
    nodes, st = await run("what does attention do")
    assert nodes.count("generate") == 2 and nodes.count("self_check") == 2
    assert planted in log["prompts"][1][0]  # the feedback reached the regeneration
    assert st["warning"].startswith("Some statements may not be supported by your material:")
    assert planted in st["warning"]
    assert [m["role"] for m in st["messages"]] == ["user", "assistant"]  # one reply despite regeneration


@pytest.mark.asyncio
async def test_regeneration_that_fixes_the_claim_clears_warning(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["Bad [1].", "Good [1]."]
    s["json"] = [{"unsupported": ["Bad [1]."]}, {"unsupported": []}]
    _, st = await run("q")
    assert st["answer"] == "Good [1]." and st["warning"] is None


@pytest.mark.asyncio
async def test_self_check_skipped_for_not_in_material(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["This isn't in your material."]
    nodes, _ = await run("q")
    assert "self_check" in nodes and log["checks"] == []  # node runs, makes no call


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [{"unsupported": "none"}, {}, {"unsupported": [3, None]}])
async def test_self_check_odd_json_counts_as_supported(script, reply):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["A [1]."]
    s["json"] = [reply]
    nodes, st = await run("q")
    assert nodes.count("generate") == 1 and st["warning"] is None


@pytest.mark.asyncio
async def test_explain_uses_explain_prompt(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["Step 1 [1]."]
    await run("explain attention")
    assert log["prompts"][0][1] == agent.EXPLAIN_SYSTEM


@pytest.mark.asyncio
async def test_connect_makes_no_llm_call(script, monkeypatch):
    s, log, run = script

    async def fake_connect(a, b):
        return {"summary": f"{a}~{b}", "html_graph": "<g/>"}

    monkeypatch.setattr(agent.query, "connect_topics", fake_connect)
    _, st = await run("connect enzymes with catalysts", thread="c")
    assert st["answer"] == "enzymes~catalysts" and st["graph"] == "<g/>" and log["prompts"] == []


@pytest.mark.asyncio
async def test_second_turn_sees_history_and_fresh_turn_fields(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)], [_doc(-6.0)], [_doc(2.0)]]
    s["complete"] = ["Bad [1].", "Bad [1].", "its complexity rewrite", "Fine [1]."]
    s["json"] = [{"unsupported": ["Bad [1]."]}, {"unsupported": ["Bad [1]."]}, {"unsupported": []}]
    _, st1 = await run("what is attention")
    assert st1["warning"]
    _, st2 = await run("what about its complexity?")
    assert len(st2["messages"]) == 4 and st2["warning"] is None and st2["retries"] == 1
    assert "what is attention" in log["prompts"][2][0]  # rewrite prompt carries prior messages


@pytest.mark.asyncio
async def test_rewrite_prompt_names_the_topic(script):
    s, log, run = script
    tid = topics.create_topic("Deep Learning")["id"]
    s["search"] = [[_doc(-6.0)], [_doc(2.0)]]
    s["complete"] = ["attention cost", "A [1]."]
    await run("why is it slow for long inputs", topic_id=tid)
    assert "Deep Learning" in log["prompts"][0][0]


@pytest.mark.asyncio
async def test_failed_turn_leaves_no_orphan_question(script, monkeypatch):
    s, log, run = script

    async def boom(q, topic_id):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(agent, "search_material", boom)
    with pytest.raises(RuntimeError):
        await run("first question")
    monkeypatch.setattr(agent.query, "connect_topics", lambda a, b: _async({"summary": f"{a}~{b}"}))
    _, st = await run("connect enzymes with catalysts")
    assert [m["content"] for m in st["messages"]] == ["connect enzymes with catalysts", "enzymes~catalysts"]


async def _async(value):
    return value


@pytest.mark.asyncio
async def test_rewrite_that_repeats_a_tried_query_stops_searching(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)]] * 3
    s["complete"] = ["vague", "vague"]
    nodes, st = await run("vague")
    assert log["queries"] == ["vague"] and nodes.count("rewrite_query") == 1
    assert st["answer"] == "This isn't in your material. Searched for: 'vague'"


# ---------------------------------------------------------------------------
# Quiz and review sessions
# ---------------------------------------------------------------------------

from core import memory, tutor  # noqa: E402
from core.db import LearnerFact, QuizAttempt, session  # noqa: E402


def _q(concept, chunk_id, kind):
    mcq = kind == "mcq"
    return tutor.Question(kind=kind, question=f"Q about {concept}?", options=["w", "x", "y", "z"] if mcq else [],
                          answer="B" if mcq else "ans", difficulty="easy", concept=concept, chunk_id=chunk_id,
                          citation={"source_title": "Notes", "page": 1, "chunk_id": chunk_id, "text": f"{concept} text",
                                    "kind": "text", "image_url": None, "url": None, "start_s": None, "score": 0.0})


@pytest.fixture
def quiz(script, monkeypatch):
    """Fake tutor targets, questions and verdicts; the real tutor.record writes attempts."""
    s, log, run = script
    f = {"targets": [(f"c{i}", i) for i in range(1, 6)], "verdicts": [], "picks": []}

    async def pick(topic_id, about=None, review=False, n=5, today=None):
        f["picks"].append({"topic_id": topic_id, "about": about, "review": review})
        return list(f["targets"])

    async def generate(concept, chunk_id, kind, facts=""):
        return _q(concept, chunk_id, kind)

    async def grade(question, answer, facts=""):
        v = f["verdicts"].pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(tutor, "pick_targets", pick)
    monkeypatch.setattr(tutor, "generate_question", generate)
    monkeypatch.setattr(tutor, "grade", grade)
    return f, run


@pytest.mark.asyncio
async def test_quiz_session_runs_to_summary(quiz):
    f, run = quiz
    _, st = await run("quiz me on enzymes")
    assert f["picks"] == [{"topic_id": None, "about": "enzymes", "review": False}]
    assert st["answer"].startswith("Question 1/5 · c1 · easy\n\nQ about c1?")
    assert "A) w\nB) x\nC) y\nD) z" in st["answer"] and st["answer"].endswith("Reply with a letter.")
    assert st["citations"][0]["chunk_id"] == 1
    f["verdicts"] = [("correct", ""), ("correct", "Good."), ("wrong", "Because."), ("correct", ""), ("correct", "")]
    _, st = await run("B")
    assert st["answer"].startswith("✅ Correct.") and "Question 2/5 · c2" in st["answer"]
    assert st["answer"].endswith("Answer in a sentence or two.")
    await run("ans")
    _, st = await run("A")
    assert st["answer"].startswith("❌ Not quite — the answer is B) x. Because.")
    await run("ans")
    _, st = await run("B")
    assert "Session done: 4/5 correct. To review: c3 (tomorrow)" in st["answer"] and st["quiz"] is None
    with session() as s:
        assert len(s.exec(select_attempts()).all()) == 5


def select_attempts():
    from sqlmodel import select
    return select(QuizAttempt)


def test_feedback_has_no_double_period():
    q = _q("enzyme", 1, "mcq").model_copy(update={"options": ["w", "It blocks the active site.", "y", "z"]})
    expected = "❌ Not quite — the answer is B) It blocks the active site. From the passage."
    assert agent._feedback(q, "wrong", "From the passage.") == expected


@pytest.mark.asyncio
async def test_stop_ends_quiz(quiz):
    f, run = quiz
    await run("quiz me")
    _, st = await run("stop")
    assert st["answer"] == "Quiz stopped after 0 questions." and st["quiz"] is None


@pytest.mark.asyncio
async def test_new_quiz_replaces_active_one(quiz):
    f, run = quiz
    await run("quiz me on enzymes")
    f["targets"] = [("weak", 9)]
    _, st = await run("review")
    assert f["picks"][-1]["review"] is True and st["quiz"]["mode"] == "review"
    assert st["answer"].startswith("Question 1/1 · weak")


@pytest.mark.asyncio
async def test_failed_answer_turn_keeps_quiz(quiz):
    f, run = quiz
    await run("quiz me")
    f["verdicts"] = [RuntimeError("ollama down"), ("correct", "")]
    with pytest.raises(RuntimeError):
        await run("B")
    _, st = await run("B")
    assert st["answer"].startswith("✅ Correct.") and "Question 2/5 · c2" in st["answer"]


@pytest.mark.asyncio
async def test_review_with_nothing_due_says_so(quiz):
    f, run = quiz
    f["targets"] = []
    _, st = await run("review")
    assert st["answer"] == agent.REVIEW_EMPTY and st["quiz"] is None


@pytest.mark.asyncio
async def test_remember_skips_quiz_turns_and_feeds_generate(quiz, script, monkeypatch):
    f, run = quiz
    s, log, _ = script
    calls = []

    async def fake_update(message):
        calls.append(message)
        return 0

    monkeypatch.setattr(memory, "update", fake_update)
    await run("quiz me")                       # quiz start: not remembered
    f["verdicts"] = [("correct", "")]
    await run("my answer is B")                # quiz answer: not remembered
    assert calls == []
    await run("stop")
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["Noted [1]."]
    await run("My exam is on 12 Dec")          # normal chat: remembered (in the background)
    await asyncio.gather(*agent._background)
    assert calls == ["My exam is on 12 Dec"]
    with session() as db:
        db.add(LearnerFact(fact="Prefers short examples"))
        db.commit()
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["Short [1]."]
    await run("what does attention do")
    assert "Prefers short examples" in log["prompts"][-1][1]


@pytest.mark.asyncio
async def test_done_when_review_targets_planted_mistakes(script, monkeypatch):
    """Planted: alpha wrong, beta partly right, gamma–epsilon right. The next review asks exactly [alpha, beta]."""
    from tests.planting import plant

    s, log, run = script
    names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    tid, _ = plant([f"{n} is a key idea in this course" for n in names],
                   {i: [(n, "is", n)] for i, n in enumerate(names)})

    async def complete_json(prompt, system=None, **kw):
        concept = next(n for n in names if f"'{n}'" in prompt or f"Q about {n}" in prompt)
        if "Grade the student's answer" in prompt:
            return {"verdict": "partial" if concept == "beta" else "correct", "feedback": "From the passage."}
        if "multiple-choice" in prompt:
            return {"question": f"Q about {concept}?", "options": ["w", "x", "y", "z"],
                    "answer": "A", "difficulty": "easy"}
        return {"question": f"Q about {concept}?", "answer": "ans", "difficulty": "easy"}

    async def complete(prompt, system=None, **kw):
        return "Because the passage says so."

    monkeypatch.setattr(llm, "complete_json", complete_json)
    monkeypatch.setattr(llm, "complete", complete)
    _, st = await run("quiz me", topic_id=tid)
    for _ in range(5):
        concept = re.search(r"Question \d/5 · (\w+)", st["answer"]).group(1)
        reply = "D" if concept == "alpha" else ("A" if st["quiz"]["current"]["kind"] == "mcq" else "an answer")
        _, st = await run(reply, topic_id=tid)
    assert "Session done: 3/5 correct." in st["answer"]
    _, st = await run("review", topic_id=tid)
    assert [c for c, _ in st["quiz"]["targets"]] == ["alpha", "beta"]
    assert st["answer"].startswith("Question 1/2 · alpha")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Stop codons end translation", "Review of the evidence says it binds"])
async def test_answers_starting_with_command_words_are_answers(quiz, answer):
    f, run = quiz
    await run("quiz me")
    f["verdicts"] = [("partial", "Close.")]
    _, st = await run(answer)
    assert st["answer"].startswith("🟡 Partly right.") and st["quiz"]["index"] == 1
    assert f["picks"][-1]["review"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["stop", "Stop.", "end quiz", "quit!"])
async def test_exact_stop_commands_still_stop(quiz, command):
    f, run = quiz
    await run("quiz me")
    _, st = await run(command)
    assert st["quiz"] is None and st["answer"].startswith("Quiz stopped")


@pytest.mark.asyncio
async def test_memory_update_does_not_delay_the_answer(script, monkeypatch):
    s, log, run = script
    release = asyncio.Event()

    async def slow_update(message):
        await release.wait()  # a slow model call
        return 1

    monkeypatch.setattr(memory, "update", slow_update)
    s["search"] = [[_doc(2.0)]]
    s["complete"] = ["ATP stores energy [1]."]
    _, st = await asyncio.wait_for(run("Can you explain to me how ATP works?"), timeout=5)
    assert st["answer"] == "ATP stores energy [1]."
    release.set()
    await asyncio.gather(*agent._background)


@pytest.mark.asyncio
async def test_failed_next_question_does_not_record_the_answer_twice(quiz, monkeypatch):
    f, run = quiz
    await run("quiz me")
    calls = {"n": 0}
    real = tutor.generate_question

    async def flaky(concept, chunk_id, kind, facts=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db hiccup")
        return await real(concept, chunk_id, kind, facts)

    monkeypatch.setattr(tutor, "generate_question", flaky)
    f["verdicts"] = [("correct", ""), ("correct", "")]
    with pytest.raises(RuntimeError):
        await run("B")
    await run("B")
    with session() as db:
        assert len(db.exec(select_attempts()).all()) == 1
