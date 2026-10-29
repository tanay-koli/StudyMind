"""
StudyMind — Agent

One LangGraph state machine decides what to do with a chat message: route by
rules (qa / explain / connect / quiz / review), retrieve, grade the passages by
reranker score, rewrite and retry when they are weak, answer with citations,
then check the answer against the cited passages.
"""

import operator
import re
from typing import Annotated, Optional, TypedDict


class StudyState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # {"role", "content"}; appended per turn
    topic_id: Optional[str]
    topics: list[str]          # two topic names for connect
    intent_hint: Optional[str]  # from the UI; input only
    intent: str                # qa | explain | connect | quiz | review
    question: str              # this turn's user message
    query: str                 # current search query (question, or a rewrite)
    queries_tried: list[str]
    found: int                 # hits the last search returned, before grading
    documents: list[dict]      # hits as dicts, so the checkpointer can serialise them
    answer: str
    citations: list[dict]
    graph: Optional[str]       # connect: topic map HTML
    retries: int
    regenerated: bool
    unsupported: list[str]
    warning: Optional[str]


LABELS = {
    "route_intent": "routing",
    "retrieve": "searching",
    "grade_documents": "checking sources",
    "rewrite_query": "rewriting query",
    "generate": "answering",
    "self_check": "verifying",
    "not_found": "nothing found",
    "connect": "connecting topics",
    "deferred": "not yet available",
    "respond": "done",
}

INTENTS = ("qa", "explain", "connect", "quiz", "review")
_QUIZ = re.compile(r"\b(quiz|test me)\b", re.I)
_REVIEW = re.compile(r"\breview\b", re.I)
_CONNECT = (
    re.compile(r"\bbetween (.+?) and (.+)", re.I),
    re.compile(r"\bconnect (.+?) (?:with|and|to) (.+)", re.I),
)
_EXPLAIN = re.compile(r"^\s*(explain|why|how does|how do|what is|what are)\b", re.I)


def classify(message: str, topics: Optional[list[str]], hint: Optional[str]) -> tuple[str, list[str]]:
    """(intent, connect topic names) by rules; a UI hint wins unless it is connect without two topics."""
    topics = [t for t in (topics or []) if t.strip()]
    if hint in INTENTS and (hint != "connect" or len(topics) >= 2):
        return hint, topics[:2] if hint == "connect" else []
    if _QUIZ.search(message):
        return "quiz", []
    if _REVIEW.search(message):
        return "review", []
    if len(topics) >= 2:
        return "connect", topics[:2]
    for pattern in _CONNECT:
        if m := pattern.search(message):
            names = [g.strip(" ?.!\t") for g in m.groups()]
            if all(names):
                return "connect", names
    if _EXPLAIN.search(message):
        return "explain", []
    return "qa", []
