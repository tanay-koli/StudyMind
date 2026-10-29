"""StudyMind — the LangGraph agent: routing, corrective retrieval, self-check, persistence (offline)."""

import pytest

from core import agent
from core.llm import llm


@pytest.mark.parametrize("msg,topics,hint,expected", [
    ("quiz me on attention", None, None, ("quiz", [])),
    ("Can you test me?", None, None, ("quiz", [])),
    ("I want to review today", None, None, ("review", [])),
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
    assert nodes == ["route_intent", "retrieve", "grade_documents", "generate", "self_check", "respond"]
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
async def test_empty_rewrite_falls_back_to_question(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)], [_doc(2.0)]]
    s["complete"] = ["   ", "A [1]."]
    await run("vague")
    assert log["queries"] == ["vague", "vague"]


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
async def test_deferred_and_connect_make_no_llm_call(script, monkeypatch):
    s, log, run = script

    async def fake_connect(a, b):
        return {"summary": f"{a}~{b}", "html_graph": "<g/>"}

    monkeypatch.setattr(agent.query, "connect_topics", fake_connect)
    _, st = await run("quiz me", thread="q")
    assert st["answer"] == "Quizzes and reviews are coming in Phase 5."
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
