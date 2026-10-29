"""
StudyMind — study guide by map-reduce over every chunk of a topic.

units() packs the topic's chunks into ~UNIT_TOKENS sections, build() summarises
each (map) then derives overview / concepts / questions from the summaries
(reduce). Sections are copied from the map step, so none can be dropped.
"""

import logging
from dataclasses import dataclass

from pydantic import BaseModel
from sqlmodel import select

from core import chunking
from core.db import Chunk, Source, session
from core.llm import llm

log = logging.getLogger("studymind.studyguide")

UNIT_TOKENS = 2500
REDUCE_TOKENS = 6000
FAILED_POINT = "(could not summarise this section)"


class Term(BaseModel):
    term: str
    definition: str


class SectionSummary(BaseModel):
    heading: str
    source_title: str
    pages: str
    key_points: list[str]


class StudyGuide(BaseModel):
    topic: str
    overview: str
    sections: list[SectionSummary]
    key_concepts: list[str]
    terms: list[Term]
    review_questions: list[str]


class _Map(BaseModel):
    key_points: list[str]
    terms: list[Term] = []


class _Reduce(BaseModel):
    overview: str = ""
    key_concepts: list[str] = []
    review_questions: list[str] = []


@dataclass
class Unit:
    source_title: str
    headings: list[str]
    pages: str
    text: str


def _pages(pages: list[int]) -> str:
    if not pages:
        return ""
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _pieces(rows: list[tuple[Chunk, int]]) -> list[list[tuple[Chunk, int]]]:
    """One piece per section; a section over budget is split by chunk."""
    pieces, cur, size = [], [], 0
    for row in rows:
        if cur and size + row[1] > UNIT_TOKENS:
            pieces.append(cur)
            cur, size = [], 0
        cur.append(row)
        size += row[1]
    return pieces + [cur]


def units(topic_id: str) -> list[Unit]:
    with session() as s:
        rows = s.exec(
            select(Chunk, Source.title).join(Source, Source.id == Chunk.source_id)
            .where(Chunk.topic_id == topic_id).order_by(Chunk.source_id, Chunk.ord)
        ).all()

    by_source: dict[int, list] = {}
    for chunk, title in rows:
        by_source.setdefault(chunk.source_id, []).append(
            (chunk, title, len(chunking.token_spans(chunk.text))))

    out: list[Unit] = []
    for items in by_source.values():
        title = items[0][1]
        groups: list[list[tuple[Chunk, int]]] = []
        prev = None
        for chunk, _, n in items:
            if not groups or chunk.section_heading != prev:
                groups.append([])
            groups[-1].append((chunk, n))
            prev = chunk.section_heading

        packed: list[list[tuple[Chunk, int]]] = []
        size = 0
        for piece in (p for g in groups for p in _pieces(g)):
            n = sum(t for _, t in piece)
            if packed and size + n <= UNIT_TOKENS:
                packed[-1] += piece
                size += n
            else:
                packed.append(list(piece))
                size = n

        for i, rows_ in enumerate(packed, 1):
            chunks = [c for c, _ in rows_]
            headings = list(dict.fromkeys(c.section_heading for c in chunks if c.section_heading))
            out.append(Unit(
                source_title=title,
                headings=headings or [f"{title} (part {i})"],
                pages=_pages([c.page for c in chunks if c.page is not None]),
                text="\n\n".join(c.text for c in chunks),
            ))
    return out


MAP_SYSTEM = "You are a study assistant. Use only the given text."


async def _map(unit: Unit) -> _Map | None:
    prompt = (
        f"Summarise this section of '{unit.source_title}' for a study guide.\n"
        'Reply with JSON: {"key_points": ["..."], "terms": [{"term": "...", "definition": "..."}]}\n'
        "Give 3-6 specific key points and the important terms defined in the text.\n\n"
        f"{unit.text}"
    )
    for _ in range(2):
        try:
            return _Map.model_validate(await llm.complete_json(prompt, system=MAP_SYSTEM, timeout=180))
        except Exception:
            log.warning("study guide map failed for '%s'", unit.source_title, exc_info=True)
    return None


async def _reduce(topic_name: str, sections: list[SectionSummary], terms: list[Term]) -> _Reduce | None:
    points = "\n\n".join(f"{s.heading}:\n" + "\n".join(f"- {p}" for p in s.key_points) for s in sections)
    glossary = "\n".join(f"- {t.term}: {t.definition}" for t in terms)
    body = f"{points}\n\nTerms:\n{glossary}"  # terms come last, so truncation drops them first
    # shortcut: reduce input truncated at 6000 tokens, upgrade to hierarchical reduce if large topics lose coverage in the overview
    spans = chunking.token_spans(body)
    if len(spans) > REDUCE_TOKENS:
        body = body[:spans[REDUCE_TOKENS - 1][1]]
    prompt = (
        f"Combine these section summaries of '{topic_name}' into a study guide.\n"
        'Reply with JSON: {"overview": "2-3 sentences", "key_concepts": ["5-8 main ideas"], '
        '"review_questions": ["3-5 questions"]}\n\n'
        f"{body}"
    )
    try:
        r = _Reduce.model_validate(await llm.complete_json(prompt, system=MAP_SYSTEM, timeout=180))
    except Exception:
        log.warning("study guide reduce failed", exc_info=True)
        return None
    return r if r.overview.strip() else None  # an empty overview is a failed reduce


async def build(topic_name: str, unit_list: list[Unit]) -> tuple[StudyGuide, bool]:
    """Returns the guide and whether it is complete (no map failure, reduce succeeded)."""
    sections, terms, seen, failed = [], [], set(), 0
    for u in unit_list:
        m = await _map(u)
        failed += m is None
        sections.append(SectionSummary(
            heading=" / ".join(u.headings), source_title=u.source_title, pages=u.pages,
            key_points=m.key_points if m else [FAILED_POINT]))
        for t in (m.terms if m else []):
            if t.term.lower() not in seen:
                seen.add(t.term.lower())
                terms.append(t)

    # Nothing to reduce when every section failed
    reduced = await _reduce(topic_name, sections, terms) if failed < len(unit_list) else None
    r = reduced or _Reduce()
    guide = StudyGuide(topic=topic_name, sections=sections, terms=terms, overview=r.overview,
                       key_concepts=r.key_concepts, review_questions=r.review_questions)
    return guide, failed == 0 and reduced is not None


def render_markdown(guide: StudyGuide) -> str:
    lines = [f"# Study Guide: {guide.topic}", "", guide.overview, "", "## Sections", ""]
    for s in guide.sections:
        where = f"{s.source_title} · p.{s.pages}" if s.pages else s.source_title
        lines += [f"### {s.heading} ({where})", *(f"- {p}" for p in s.key_points), ""]
    lines += ["## Key Concepts", *(f"- {c}" for c in guide.key_concepts), "",
              "## Terms", *(f"- **{t.term}**: {t.definition}" for t in guide.terms), "",
              "## Review Questions", *(f"{i}. {q}" for i, q in enumerate(guide.review_questions, 1))]
    return "\n".join(lines)
