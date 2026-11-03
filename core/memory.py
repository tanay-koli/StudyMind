"""
StudyMind — Learner memory

Free-form facts about the learner (goals, exam date, preferred explanation style).
The LLM proposes ADD / UPDATE / DELETE operations against the current facts
(Mem0's pattern); the facts are added to the tutor's prompts.
"""

import logging
import re
from datetime import datetime

from sqlmodel import select

from core.db import LearnerFact, session
from core.llm import llm

log = logging.getLogger("studymind.memory")

MAX_FACTS = 20
_PERSONAL = re.compile(r"\b(i|i'm|im|my|me|i've|i'll)\b|\bexam\b", re.I)
MEMORY_SYSTEM = "You keep a short list of durable facts about a student: goals, deadlines, preferences, background."


def should_remember(message: str) -> bool:
    """Cheap gate: only messages that talk about the learner go to the LLM."""
    return bool(_PERSONAL.search(message))


def facts() -> list[LearnerFact]:
    with session() as s:
        return list(s.exec(select(LearnerFact).order_by(LearnerFact.id)).all())


def profile() -> str:
    lines = [f"- {f.fact}" for f in facts()]
    return "About the student:\n" + "\n".join(lines) if lines else ""


def delete(fact_id: int) -> bool:
    with session() as s:
        row = s.get(LearnerFact, fact_id)
        if row is None:
            return False
        s.delete(row)
        s.commit()
    return True


async def update(message: str) -> int:
    """Apply the LLM's ADD / UPDATE / DELETE ops for this message; returns how many were applied."""
    current = facts()
    listing = "\n".join(f"[{f.id}] {f.fact}" for f in current) or "(none)"
    try:
        reply = await llm.complete_json(
            f"Known facts:\n{listing}\n\nThe student said: {message}\n\n"
            "Return the changes to the known facts. Keep only durable facts about the student, not about the "
            'subject. Reply with JSON only: {"ops": [{"op": "ADD", "fact": "..."}, '
            '{"op": "UPDATE", "id": 1, "fact": "..."}, {"op": "DELETE", "id": 2}]} (an empty list if nothing changes).',
            system=MEMORY_SYSTEM)
    except Exception as e:
        log.warning("learner memory update failed: %s", e)
        return 0
    ops = reply.get("ops") if isinstance(reply.get("ops"), list) else []
    applied = 0
    now = datetime.now().isoformat()
    with session() as s:
        count = len(current)
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind, fact = op.get("op"), str(op.get("fact") or "").strip()
            row = s.get(LearnerFact, op["id"]) if isinstance(op.get("id"), int) else None
            if kind == "ADD" and fact and count < MAX_FACTS:
                s.add(LearnerFact(fact=fact, updated_at=now))
                count += 1
            elif kind == "UPDATE" and row is not None and fact:
                row.fact, row.updated_at = fact, now
                s.add(row)
            elif kind == "DELETE" and row is not None:
                s.delete(row)
                count -= 1
            else:
                continue
            applied += 1
        s.commit()
    return applied
