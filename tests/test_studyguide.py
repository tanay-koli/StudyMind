"""StudyMind — study guide map-reduce (offline, fake LLM)."""

import json

import pytest

from core import studyguide
from core.db import Chunk, Source, session
from core.studyguide import FAILED_POINT, Unit, build, render_markdown, units


def seed(chunks, title="Book", topic_id="general"):
    """chunks: [(heading, page, text)] in one source."""
    with session() as s:
        src = Source(topic_id=topic_id, source_type="note", title=title)
        s.add(src)
        s.commit()
        for i, (heading, page, text) in enumerate(chunks):
            s.add(Chunk(source_id=src.id, topic_id=topic_id, chunker="t", ord=i,
                        page=page, section_heading=heading, text=text))
        s.commit()


def test_units_group_by_section_and_pack(monkeypatch):
    seed([("A", 1, "a1 a2"), ("A", 2, "a3"), ("B", 3, "b1 b2"), ("C", 5, "c1")])
    [u] = units("general")
    assert u.headings == ["A", "B", "C"] and u.pages == "1-5" and u.source_title == "Book"

    monkeypatch.setattr(studyguide, "UNIT_TOKENS", 3)
    assert [u.headings for u in units("general")] == [["A"], ["B", "C"]]


def test_units_split_oversized_section(monkeypatch):
    monkeypatch.setattr(studyguide, "UNIT_TOKENS", 5)
    seed([("Big", None, "w1 w2 w3"), ("Big", None, "w4 w5 w6"), ("Big", None, "w7 w8")])
    got = units("general")
    assert len(got) == 2
    assert all(len(u.text.split()) <= 5 for u in got) and all(u.headings == ["Big"] for u in got)
    assert got[0].pages == ""


def test_units_headingless_windows():
    seed([(None, 3, "x y"), (None, 3, "z")], title="Notes")
    [u] = units("general")
    assert u.headings == ["Notes (part 1)"] and u.pages == "3"


def _fake(monkeypatch, map_reply, reduce_reply=None):
    from core.llm import llm
    calls = []

    async def complete(prompt, system=None, **kw):
        calls.append(prompt)
        if prompt.startswith("Combine"):
            return json.dumps(reduce_reply if reduce_reply is not None else {})
        return map_reply(prompt, sum(1 for c in calls if not c.startswith("Combine")))

    monkeypatch.setattr(llm, "complete", complete)
    return calls


GOOD = json.dumps({"key_points": ["kp"], "terms": [{"term": "T", "definition": "d"}]})
UNIT_LIST = [Unit("Book", ["A"], "1", "text a"), Unit("Book", ["B"], "2", "text b")]


@pytest.mark.asyncio
async def test_every_section_present_even_if_reduce_omits_it(monkeypatch):
    _fake(monkeypatch, lambda p, n: GOOD, {"overview": "o", "key_concepts": ["k"], "review_questions": ["q"]})
    guide, complete = await build("Topic", UNIT_LIST)
    assert [s.heading for s in guide.sections] == ["A", "B"] and complete is True
    assert guide.overview == "o" and [t.term for t in guide.terms] == ["T"]  # de-duplicated


@pytest.mark.asyncio
async def test_map_retries_once_then_marks_section(monkeypatch):
    calls = _fake(monkeypatch, lambda p, n: "not json" if "text a" in p else GOOD, {})
    guide, complete = await build("Topic", UNIT_LIST)
    assert guide.sections[0].key_points == [FAILED_POINT] and guide.sections[1].key_points == ["kp"]
    assert complete is False
    assert sum("text a" in c for c in calls) == 2


def test_render_markdown_contains_sections():
    from core.studyguide import SectionSummary, StudyGuide, Term
    md = render_markdown(StudyGuide(
        topic="Topic", overview="Over", sections=[SectionSummary(heading="A", source_title="Book", pages="1-2", key_points=["kp"])],
        key_concepts=["kc"], terms=[Term(term="T", definition="d")], review_questions=["rq?"]))
    assert md.startswith("# Study Guide: Topic")
    for part in ("Over", "## Sections", "### A (Book · p.1-2)", "- kp", "## Key Concepts", "## Terms", "## Review Questions"):
        assert part in md
