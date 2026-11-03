"""
StudyMind — Tutor

Quiz questions from the learner's material, graded answers, and SM-2 spaced
repetition per concept (concepts come from the Phase 2 concept triples).
"""

import re
from collections import Counter
from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel
from sqlmodel import select

from core import concepts, query
from core.db import ConceptEdge, ConceptMastery, QuizAttempt, session
from core.llm import llm
from core.retrieval import Hit, _hydrate, get_retriever, relevant

SESSION_SIZE = 5
QUALITY = {"correct": 5, "partial": 3, "wrong": 1}
MIN_EASE = 1.3


class Question(BaseModel):
    kind: Literal["mcq", "short"]
    question: str
    options: list[str] = []  # 4 for mcq, shown as A–D
    answer: str  # mcq: the right option's letter; short: the expected answer
    difficulty: Literal["easy", "medium", "hard"]
    concept: str
    chunk_id: int
    citation: dict


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

async def pick_targets(topic_id: Optional[str], about: Optional[str] = None, review: bool = False,
                       n: int = SESSION_SIZE, today: Optional[date] = None) -> list[tuple[str, int]]:
    """(concept, chunk_id) pairs to quiz on.

    Quiz: concepts of the passages retrieved for `about` (else the topic name), in rank order.
    Review: concepts that are due or were last answered wrong / partly, weakest first; [] if none.
    """
    if review:
        today_s = (today or date.today()).isoformat()
        with session() as s:
            rows = s.exec(select(ConceptMastery).where(
                (ConceptMastery.next_review <= today_s) | ConceptMastery.last_verdict.in_(["wrong", "partial"]))).all()
        targets = []
        for row in sorted(rows, key=lambda r: (r.next_review, r.ease, -r.wrong, r.concept)):
            if chunks := concepts.chunks_for_concept(row.concept, topic_id):
                targets.append((row.concept, chunks[0]))
        return targets[:n]

    if about is None:
        # No subject named: passages rarely contain their topic's name, so take the topic's own concepts
        return _topic_targets(topic_id, n)
    hits = relevant(await get_retriever(rewrite=False).retrieve(about, topic_id, k=12))
    tags = concepts.chunk_concepts([h.chunk_id for h in hits])
    targets, seen = [], set()
    for h in hits:
        for c in tags.get(h.chunk_id, []):
            if c not in seen:
                seen.add(c)
                targets.append((c, h.chunk_id))
    return targets[:n]


def _topic_targets(topic_id: Optional[str], n: int) -> list[tuple[str, int]]:
    """The scope's concepts: never quizzed first (most mentioned first), then the weakest; each with its best chunk."""
    q = select(ConceptEdge).order_by(ConceptEdge.id)
    if topic_id:
        q = q.where(ConceptEdge.topic_id == topic_id)
    with session() as s:
        edges = s.exec(q).all()
        mentions: dict[str, Counter] = {}
        for e in edges:
            for c in (e.head_norm, e.tail_norm):
                mentions.setdefault(c, Counter())[e.chunk_id] += 1
        known = {m.concept: m for m in s.exec(select(ConceptMastery).where(ConceptMastery.concept.in_(list(mentions))))}
    order = {c: i for i, c in enumerate(mentions)}
    ranked = sorted(mentions, key=lambda c: (c in known, known[c].ease if c in known else 0,
                                             -sum(mentions[c].values()), order[c]))
    return [(c, mentions[c].most_common(1)[0][0]) for c in ranked[:n]]


# ---------------------------------------------------------------------------
# Questions and grading
# ---------------------------------------------------------------------------

QUESTION_SYSTEM = "You write quiz questions for a student from one passage of their study material. Use only the passage."
TUTOR_SYSTEM = "You are a patient tutor. Use only the passage you are given."
UNGRADED = "Couldn't grade this answer."
_LETTERS = "ABCD"
_SHAPES = {
    "mcq": ('a multiple-choice question with 4 options and exactly one right answer',
            '{"question": "...", "options": ["...", "...", "...", "..."], "answer": "A|B|C|D", '
            '"difficulty": "easy|medium|hard"}'),
    "short": ("a short-answer question someone can answer in a sentence or two",
              '{"question": "...", "answer": "the expected answer", "difficulty": "easy|medium|hard"}'),
}


def _system(base: str, facts: str) -> str:
    return f"{base}\n{facts}" if facts else base


def _citation(chunk_id: int) -> Optional[dict]:
    rows = _hydrate([chunk_id])
    if not rows:
        return None
    c, title = rows[0]
    return query._context([Hit(c.id, c.source_id, title, c.page, c.section_heading, c.text, 0.0)])[1][0]


async def generate_question(concept: str, chunk_id: int, kind: str, facts: str = "") -> Optional[Question]:
    """One question about the concept from the chunk; None after one retry on an invalid reply."""
    citation = _citation(chunk_id)
    if citation is None:
        return None
    what, shape = _SHAPES[kind]
    prompt = (f"Passage:\n{citation['text']}\n\nWrite {what} that tests '{concept}'. "
              f"Reply with JSON only: {shape}")
    for _ in range(2):
        try:
            reply = await llm.complete_json(prompt, system=_system(QUESTION_SYSTEM, facts))
            options = [re.sub(r"^\s*[A-Da-d][).:]\s*", "", str(o)).strip() for o in reply.get("options", [])]
            q = Question(kind=kind, question=str(reply["question"]).strip(), options=options if kind == "mcq" else [],
                         answer=str(reply["answer"]).strip(), difficulty=reply.get("difficulty", "medium"),
                         concept=concept, chunk_id=chunk_id, citation=citation)
        except (ValueError, KeyError, TypeError):  # a bad reply (pydantic errors are ValueErrors); outages propagate
            continue
        if q.question and q.answer and (kind == "short" or (
                len(q.options) == 4 and all(q.options) and q.answer.upper() in _LETTERS)):
            if kind == "mcq":
                q.answer = q.answer.upper()
            return q
    return None


def _mcq_letter(answer: str, options: list[str]) -> Optional[str]:
    """The option chosen: a one-letter reply, "d) zinc", the option's text, else the first standalone uppercase A–D."""
    bare = answer.strip().strip("().").strip()
    if len(bare) == 1 and bare.upper() in _LETTERS:
        return bare.upper()
    if m := re.match(r"\s*\(?([a-dA-D])[).:]", answer):
        return m.group(1).upper()
    for letter, option in zip(_LETTERS, options):
        if bare.casefold() == option.strip().casefold():
            return letter  # before the capital-letter rule: "Vitamin C" is an option's text, not letter C
    for m in re.finditer(r"\b([A-D])\b", answer):
        if m.group(1) == "A" and not answer[:m.start()].strip() and re.match(r"\s+[a-z]", answer[m.end():]):
            continue  # "A cell divides…": the article that starts a sentence, not option A
        return m.group(1)
    return None


async def grade(question: Question, answer: str, facts: str = "") -> tuple[Optional[str], str]:
    """(verdict, feedback); verdict None when the answer couldn't be graded. Uses the stored citation text."""
    passage = question.citation["text"]
    if question.kind == "mcq":
        if _mcq_letter(answer, question.options) == question.answer:
            return "correct", ""
        right = f"{question.answer}) {question.options[_LETTERS.index(question.answer)]}"
        why = await llm.complete(
            f"Passage:\n{passage}\n\nQuestion: {question.question}\nThe right answer is {right}.\n"
            "In at most 2 sentences, explain from the passage why it is right.",
            system=_system(TUTOR_SYSTEM, facts))
        return "wrong", why.strip()
    prompt = (f"Passage:\n{passage}\n\nQuestion: {question.question}\nExpected answer: {question.answer}\n"
              f"Student's answer: {answer}\n\nGrade the student's answer against the passage. "
              'Reply with JSON only: {"verdict": "correct|partial|wrong", "feedback": "one line, from the passage"}')
    for _ in range(2):
        try:
            reply = await llm.complete_json(prompt, system=_system(TUTOR_SYSTEM, facts))
        except ValueError:  # no JSON in the reply; outages propagate so the agent keeps the quiz
            continue
        if reply.get("verdict") in QUALITY:
            return reply["verdict"], str(reply.get("feedback", "")).strip()
    return None, UNGRADED


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def sm2(row: ConceptMastery, verdict: str, today: date) -> ConceptMastery:
    """SM-2 update for one graded answer (correct 5, partial 3, wrong 1)."""
    q = QUALITY[verdict]
    if q < 3:
        row.repetitions, row.interval_days = 0, 1
    else:
        row.repetitions += 1
        # the interval uses the ease from before this answer
        row.interval_days = {1: 1, 2: 6}.get(row.repetitions) or round(row.interval_days * row.ease)
    row.ease = max(MIN_EASE, row.ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    row.next_review = (today + timedelta(days=row.interval_days)).isoformat()
    row.last_verdict = verdict
    row.attempts += 1
    row.wrong += verdict != "correct"
    return row


def record(question: Question, answer: str, verdict: Optional[str], feedback: str,
           topic_id: Optional[str], today: Optional[date] = None) -> None:
    """Log the attempt; a graded one also moves the concept's SM-2 schedule."""
    today = today or date.today()
    with session() as s:
        s.add(QuizAttempt(topic_id=topic_id, concept=question.concept, chunk_id=question.chunk_id,
                          kind=question.kind, question=question.question, answer=answer,
                          verdict=verdict, feedback=feedback))
        if verdict is not None:
            row = s.get(ConceptMastery, question.concept) or ConceptMastery(
                concept=question.concept, next_review=today.isoformat())
            s.add(sm2(row, verdict, today))
        s.commit()
