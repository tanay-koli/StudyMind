"""StudyMind — eval judge: parsing, retry, cache, calibration and agreement."""

import json

import pytest

from core.llm import llm
from eval import judge

HIT = {"source_title": "ODS", "page": 1, "chunk_id": 1, "kind": "text", "score": 1.0, "text": "heaps are trees"}
DATASET = {"q1": {"id": "q1", "type": "text", "gold_answer": "trees"},
           "q2": {"id": "q2", "type": "unanswerable", "gold_answer": "Not in the material."}}


@pytest.fixture
def judge_replies(monkeypatch):
    calls = []
    replies = []

    async def fake(prompt, system=None, **kw):
        calls.append((prompt, system, kw))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm, "complete_json", fake)

    def set_replies(rs):
        replies[:] = rs
        return calls

    return set_replies


def _row(id, answer, refused, type="text"):
    return {"id": id, "type": type, "question": "what are heaps", "hits": [HIT], "answer": answer,
            "refused": refused, "error": None}


@pytest.fixture
def result_file(tmp_path):
    p = tmp_path / "E.json"
    p.write_text(json.dumps({"config": {"name": "E"}, "rows": [
        _row("q1", "Heaps are trees [1].", False),
        _row("q2", "This isn't in your material.", True, "unanswerable")]}), encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_faithfulness_and_correctness_parse(judge_replies):
    judge_replies([{"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]},
                   {"verdict": "partial"}])
    assert await judge.faithfulness("[1] x", "a. b.") == 0.5
    assert await judge.correctness("q", "gold", "ans") == 0.5


@pytest.mark.asyncio
async def test_retry_then_null(judge_replies):
    calls = judge_replies([ValueError("no json"), ValueError("no json")])
    assert await judge.correctness("q", "g", "a") is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_judge_uses_judge_model(judge_replies, monkeypatch):
    monkeypatch.setattr(judge.settings, "judge_model", "anthropic/claude-x")
    calls = judge_replies([{"verdict": "correct"}])
    await judge.correctness("q", "g", "a")
    assert calls[0][2]["model"] == "anthropic/claude-x"


@pytest.mark.asyncio
async def test_judge_file_uses_cache_and_skips_refusals(tmp_path, judge_replies, result_file):
    calls = judge_replies([{"claims": []}, {"verdict": "correct"}])
    stats = await judge.judge_file(result_file, DATASET, tmp_path / "cache.json")
    rows = json.loads(result_file.read_text(encoding="utf-8"))["rows"]
    assert rows[0]["judge"] == {"faithfulness": 1.0, "correctness": 1.0}
    assert rows[1]["judge"] == {"faithfulness": None, "correctness": None}
    assert stats == {"judged": 1, "failures": 0}
    await judge.judge_file(result_file, DATASET, tmp_path / "cache.json")
    assert len(calls) == 2  # second pass fully cached


def _judged(n):
    return {"E": {"rows": [{**_row(f"q{i}", "x [1].", False), "judge": {"faithfulness": 1.0, "correctness": 1.0}}
                           for i in range(1, n + 1)]}}


def test_calibrate_writes_each_label_immediately(tmp_path):
    answers = iter(["c", "y", "w"])

    def ask(_):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt

    labels = tmp_path / "labels.jsonl"
    dataset = {f"q{i}": {"id": f"q{i}", "type": "text", "gold_answer": "g"} for i in range(1, 4)}
    with pytest.raises(KeyboardInterrupt):
        judge.calibrate(_judged(3), dataset, labels, n=3, ask=ask)
    lines = labels.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["correct"] == "c" and json.loads(lines[0])["faithful"] is True
    assert len(judge.sample(_judged(3), {("E", json.loads(lines[0])["id"])}, n=30)) == 2  # resume skips it


def test_agreement():
    results = {"E": {"rows": [{"id": "q1", "judge": {"correctness": 1.0, "faithfulness": 1.0}},
                              {"id": "q2", "judge": {"correctness": 0.0, "faithfulness": 0.5}}]}}
    labels = [{"config": "E", "id": "q1", "correct": "c", "faithful": True},
              {"config": "E", "id": "q2", "correct": "c", "faithful": False}]
    a = judge.agreement(labels, results)
    assert a["n"] == 2 and a["correctness"]["agree"] == 0.5 and a["faithfulness"]["agree"] == 1.0


@pytest.mark.asyncio
async def test_refused_answerable_row_is_not_a_judge_failure(tmp_path, judge_replies):
    p = tmp_path / "E.json"
    p.write_text(json.dumps({"rows": [_row("q1", "This isn't in your material.", True)]}), encoding="utf-8")
    judge_replies([{"verdict": "wrong"}])
    assert await judge.judge_file(p, DATASET, tmp_path / "c.json") == {"judged": 1, "failures": 0}
