"""StudyMind — eval labelling CLI: quote checks, propose and find."""

import json

import pytest

from core.llm import llm
from eval import label
from tests.planting import plant

TEXTS = ["Heaps support insert in log n time using sift up.", "Skiplists are layered linked lists with express lanes."]


def _answers(seq):
    it = iter(seq)
    return lambda _prompt: next(it)


def _fake_drafts(monkeypatch, quote_for=lambda text: text.split(" using")[0].split(" with")[0]):
    async def fake(prompt, system=None, **kw):
        text = next(t for t in TEXTS if t in prompt)
        return {"question": f"Q about {text[:6]}?", "gold_answer": "g", "quote": quote_for(text)}

    monkeypatch.setattr(llm, "complete_json", fake)


def test_next_id_and_first_quote():
    assert label.next_id([]) == "q001" and label.next_id([{"id": "q009"}, {"id": "q002"}]) == "q010"
    assert (label.first_quote("Hi. A heap is a tree with the heap property at every node.")
            == "A heap is a tree with the heap property at every node.")
    assert label.first_quote("Too short.") is None


@pytest.mark.asyncio
async def test_draft_rejects_non_verbatim_quote_after_one_retry(monkeypatch):
    calls = []

    async def fake(prompt, system=None, **kw):
        calls.append(prompt)
        return {"question": "q", "gold_answer": "g", "quote": "invented words here now"}

    monkeypatch.setattr(llm, "complete_json", fake)
    assert await label.draft(TEXTS[0], "text") is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_propose_accept_skip_appends(tmp_path, monkeypatch):
    tid, _ = plant(TEXTS, {}, topic="ODS")
    _fake_drafts(monkeypatch)
    path = tmp_path / "d.jsonl"
    n = await label.propose(2, "text", path, ask=_answers(["a", "s"]))
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert n == 1 and len(rows) == 1
    row = rows[0]
    assert row["id"] == "q001" and row["type"] == "text" and row["topic_id"] == tid
    assert row["gold"][0]["source"] == "ODS notes" and row["gold"][0]["quote"] in " ".join(TEXTS)


@pytest.mark.asyncio
async def test_propose_skips_chunks_already_in_the_dataset(tmp_path, monkeypatch):
    plant(TEXTS, {}, topic="ODS")
    _fake_drafts(monkeypatch)
    path = tmp_path / "d.jsonl"
    await label.propose(1, "text", path, ask=_answers(["a"]))
    await label.propose(1, "text", path, ask=_answers(["a"]))
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q001", "q002"]
    assert rows[0]["gold"][0]["page"] != rows[1]["gold"][0]["page"]


@pytest.mark.asyncio
async def test_find_ticks_gold_and_unanswerable_has_empty_gold(tmp_path):
    tid, _ = plant(TEXTS, {}, topic="ODS")
    path = tmp_path / "d.jsonl"
    row = await label.find("heaps insert sift up", "text", path, ask=_answers(["1", "log n"]), topic_id=tid)
    assert row["gold"] == [{"source": "ODS notes", "page": 1, "quote": TEXTS[0]}] and row["gold_answer"] == "log n"
    row = await label.find("who invented treaps", "unanswerable", path, ask=_answers(["", ""]), topic_id=tid)
    assert row["gold"] == [] and row["gold_answer"] == "Not in the material." and row["id"] == "q002"
