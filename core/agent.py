"""
StudyMind — Agent

One LangGraph state machine decides what to do with a chat message: route by
rules (qa / explain / connect / quiz / review), retrieve, grade the passages by
reranker score, rewrite and retry when they are weak, answer with citations,
then check the answer against the cited passages. Quiz and review sessions run
one question per turn and survive between turns in the checkpointed state.
"""

import asyncio
import dataclasses
import logging
from datetime import date, timedelta
import operator
import re
from contextlib import asynccontextmanager
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from core import memory, query, tutor
from core.db import ConceptMastery, session
from core.config import settings
from core.llm import llm
from core.retrieval import Hit, get_retriever, relevant

log = logging.getLogger("studymind.agent")

MAX_RETRIES = 2
NOT_FOUND = "This isn't in your material."
REVIEW_EMPTY = "Nothing to review yet — answer some quiz questions first, or try 'quiz me on <topic>'."
QUIZ_EMPTY = "I couldn't find quizzable material for that yet."
NO_QUESTIONS = "I couldn't write questions from this material."
WARNING = "Some statements may not be supported by your material:"
EXPLAIN_SYSTEM = (
    query.CITE_RULES
    + "Explain step by step for a student, defining terms before using them.\n"
    + "If the context does not contain the answer, reply exactly: This isn't in your material."
)
REWRITE_SYSTEM = "You turn a student's question into one short, specific search query for their study notes."
CHECK_SYSTEM = "You check an answer against the numbered passages it cites."


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
    # Survives between turns (route_intent does not reset it): {mode, topic_id, targets: [[concept, chunk_id]],
    # index, current: Question dict, results: [[concept, verdict]]}
    quiz: Optional[dict]


LABELS = {
    "route_intent": "routing",
    "retrieve": "searching",
    "grade_documents": "checking sources",
    "rewrite_query": "rewriting query",
    "generate": "answering",
    "self_check": "verifying",
    "not_found": "nothing found",
    "connect": "connecting topics",
    "quiz_start": "making a question",
    "quiz_answer": "grading",
    "quiz_stop": "stopping quiz",
    "remember": "remembering",
    "respond": "done",
}

INTENTS = ("qa", "explain", "connect", "quiz", "review")
# Commands only: "quiz" or "review" inside a question ("the peer review process") stays a question
_QUIZ = re.compile(r"^\s*quiz\b|\b(quiz|test) me\b", re.I)
_REVIEW = re.compile(r"^\s*review\b|\breview (me|my)\b", re.I)
_CONNECT = (
    # "the difference between TCP and UDP" is a question, not a topic map
    re.compile(r"\b(?:connections?|links?|relationships?|relations?)\s+between (.+?) and (.+)", re.I),
    re.compile(r"\bconnect (.+?) (?:with|and|to) (.+)", re.I),
)
_EXPLAIN = re.compile(r"^\s*(explain|why|how does|how do|what is|what are)\b", re.I)
# During a quiz only whole-message commands count: "Stop codons end translation" is an answer
QUIZ_STOP = re.compile(r"^\s*(stop|end( the)? quiz|quit)\s*[.!]?\s*$", re.I)
_NEW_SESSION = re.compile(r"^\s*(review( my weak (topics|concepts))?|(quiz|test) me( on .+)?)\s*[.!?]?\s*$", re.I)


def _quiz_about(message: str) -> Optional[str]:
    """'quiz me on enzymes' → 'enzymes'."""
    m = re.search(r"\b(?:on|about)\s+(.+)", message, re.I)
    return (m.group(1).strip(" ?.!\t") or None) if m else None


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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def search_material(query_text: str, topic_id: Optional[str]) -> list[dict]:
    """Reranked passages for the query; the agent rewrites itself, so never the rewriting retriever."""
    hits = await get_retriever(rewrite=False).retrieve(query_text, topic_id)
    return [dataclasses.asdict(h) for h in hits]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def route_intent(state: StudyState) -> dict:
    question = state["question"]
    if state.get("quiz") and QUIZ_STOP.match(question):
        intent, topics = "quiz_stop", []
    elif state.get("quiz") and not _NEW_SESSION.match(question):
        intent, topics = "quiz_answer", []  # a new quiz/review command replaces the session instead
    else:
        intent, topics = classify(question, state.get("topics"), state.get("intent_hint"))
    return {
        "intent": intent, "topics": topics, "query": question,
        # per-turn fields start fresh; messages carry the conversation (respond saves this turn)
        "queries_tried": [], "found": 0, "documents": [], "answer": "", "citations": [], "graph": None,
        "retries": 0, "regenerated": False, "unsupported": [], "warning": None,
    }


async def retrieve(state: StudyState) -> dict:
    docs = await search_material(state["query"], state.get("topic_id"))
    return {"documents": docs, "found": len(docs), "queries_tried": state["queries_tried"] + [state["query"]]}


async def grade_documents(state: StudyState) -> dict:
    return {"documents": relevant(state["documents"])}


async def rewrite_query(state: StudyState) -> dict:
    history = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"][-4:])
    tried = ", ".join(repr(q) for q in state["queries_tried"])
    # Without the topic, a 7B model rewrites "why is it slow" toward whatever it guesses
    topic = f"The student is studying: {query._topic_name(state['topic_id'])}\n" if state.get("topic_id") else ""
    reply = await llm.complete(
        f"{topic}Recent conversation:\n{history}\n\nQuestion: {state['question']}\n"
        f"Searches that found nothing relevant: {tried}\n\n"
        "Write one short, specific search query that would find the answer. Output only the query.",
        system=REWRITE_SYSTEM,
    )
    lines = reply.strip().splitlines()
    new = lines[0].strip(" \"'") if lines else ""
    if not new or new in state["queries_tried"]:
        new = state["question"]
    return {"query": new, "retries": state["retries"] + 1}


async def generate(state: StudyState) -> dict:
    context, citations = query._context([Hit(**d) for d in state["documents"]])
    prompt = f"{context}\n\nQuestion: {state['question']}"
    if state["unsupported"]:
        prompt += ("\n\nThese sentences were not supported by the passages; drop or fix them:\n"
                   + "\n".join(f"- {u}" for u in state["unsupported"]))
    system = EXPLAIN_SYSTEM if state["intent"] == "explain" else query.ANSWER_SYSTEM
    if profile := memory.profile():
        system += "\n" + profile
    answer = await llm.complete(prompt, system=system)
    # Called with flagged sentences means this is the one regeneration
    return {"answer": answer, "citations": citations, "regenerated": bool(state["unsupported"])}


async def self_check(state: StudyState) -> dict:
    if state["answer"].strip().startswith(NOT_FOUND):
        return {"unsupported": []}
    context, _ = query._context([Hit(**d) for d in state["documents"]])
    try:
        reply = await llm.complete_json(
            f"{context}\n\nAnswer:\n{state['answer']}\n\n"
            "List every sentence of the answer that the numbered passages do not support. "
            'Reply with JSON: {"unsupported": ["sentence", ...]} (an empty list if all are supported).',
            system=CHECK_SYSTEM,
        )
    except ValueError:
        log.warning("self_check reply had no JSON; treating the answer as supported", exc_info=True)
        return {"unsupported": []}
    flagged = reply.get("unsupported")
    flagged = [x for x in flagged if isinstance(x, str) and x.strip()] if isinstance(flagged, list) else []
    return {"unsupported": flagged}


async def not_found(state: StudyState) -> dict:
    tried = ", ".join(repr(q) for q in state["queries_tried"])
    return {"answer": f"{NOT_FOUND} Searched for: {tried}" if tried else NOT_FOUND, "citations": []}


async def connect(state: StudyState) -> dict:
    res = await query.connect_topics(*state["topics"][:2])
    return {"answer": res["summary"], "graph": res.get("html_graph")}


def _format_question(q: tutor.Question, i: int, total: int) -> str:
    text = f"Question {i}/{total} · {q.concept} · {q.difficulty}\n\n{q.question}"
    if q.kind == "mcq":
        options = "\n".join(f"{letter}) {option}" for letter, option in zip("ABCD", q.options))
        return f"{text}\n\n{options}\n\nReply with a letter."
    return f"{text}\n\nAnswer in a sentence or two."


async def _next_question(targets: list, index: int, facts: str) -> tuple[Optional[tutor.Question], list]:
    """The question for targets[index]; targets that fail to produce one are dropped."""
    targets = list(targets)
    while index < len(targets):
        concept, chunk_id = targets[index]
        q = await tutor.generate_question(concept, chunk_id, "mcq" if index % 2 == 0 else "short", facts)
        if q is not None:
            return q, targets
        targets.pop(index)
    return None, targets


async def quiz_start(state: StudyState) -> dict:
    review = state["intent"] == "review"
    about = None if review else _quiz_about(state["question"])
    targets = await tutor.pick_targets(state.get("topic_id"), about=about, review=review)
    if not targets:
        return {"answer": REVIEW_EMPTY if review else QUIZ_EMPTY, "quiz": None}
    q, targets = await _next_question([list(t) for t in targets], 0, memory.profile())
    if q is None:
        return {"answer": NO_QUESTIONS, "quiz": None}
    quiz = {"mode": "review" if review else "quiz", "topic_id": state.get("topic_id"), "targets": targets,
            "index": 0, "current": q.model_dump(), "results": []}
    return {"answer": _format_question(q, 1, len(targets)), "citations": [q.citation], "quiz": quiz}


def _feedback(q: tutor.Question, verdict: Optional[str], feedback: str) -> str:
    if verdict == "correct":
        return "✅ Correct."
    if verdict == "partial":
        return f"🟡 Partly right. {feedback}".strip()
    if verdict == "wrong":
        right = f"{q.answer}) {q.options['ABCD'.index(q.answer)]}" if q.kind == "mcq" else q.answer
        return f"❌ Not quite — the answer is {right.rstrip('.')}. {feedback}".strip()
    return "⚠️ Couldn't grade this answer."


def _summary(results: list) -> str:
    correct = sum(v == "correct" for _, v in results)
    weak = [c for c, v in results if v in ("wrong", "partial")]
    text = f"Session done: {correct}/{len(results)} correct."
    if weak:
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        with session() as s:
            due = {c: getattr(s.get(ConceptMastery, c), "next_review", tomorrow) for c in weak}
        text += " To review: " + ", ".join(f"{c} ({'tomorrow' if due[c] == tomorrow else due[c]})" for c in weak)
    return text


async def quiz_answer(state: StudyState) -> dict:
    quiz = state["quiz"]
    q = tutor.Question(**quiz["current"])
    facts = memory.profile()
    verdict, feedback = await tutor.grade(q, state["question"], facts)
    results = quiz["results"] + [[q.concept, verdict]]
    line = _feedback(q, verdict, feedback)
    nq, targets = await _next_question(quiz["targets"], quiz["index"] + 1, facts)
    # Recorded only once the turn can't fail any more: a retried answer must not move SM-2 twice
    tutor.record(q, state["question"], verdict, feedback, quiz["topic_id"])
    if nq is None:
        return {"answer": f"{line}\n\n{_summary(results)}", "citations": [q.citation], "quiz": None}
    nxt = dict(quiz, targets=targets, index=quiz["index"] + 1, current=nq.model_dump(), results=results)
    return {"answer": f"{line}\n\n{_format_question(nq, nxt['index'] + 1, len(targets))}",
            "citations": [q.citation, nq.citation], "quiz": nxt}


async def quiz_stop(state: StudyState) -> dict:
    return {"answer": f"Quiz stopped after {len(state['quiz']['results'])} questions.", "quiz": None}


async def respond(state: StudyState) -> dict:
    warning = f"{WARNING} " + "; ".join(state["unsupported"]) if state["unsupported"] else None
    # Question and answer are saved together, so an interrupted turn leaves no unanswered question
    return {"messages": [{"role": "user", "content": state["question"]},
                         {"role": "assistant", "content": state["answer"]}], "warning": warning}


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

_background: set[asyncio.Task] = set()  # memory updates in flight (a reference keeps them from being collected)


async def _remember_later(message: str) -> None:
    try:
        await memory.update(message)
    except Exception:
        log.warning("learner memory update failed", exc_info=True)


async def remember(state: StudyState) -> dict:
    # Personal statements arrive as ordinary chat; quiz turns ("quiz me", "my answer is B") never update memory.
    # The update is a model call: run it in the background so the answer isn't held back by it
    if state["intent"] in ("qa", "explain") and memory.should_remember(state["question"]):
        task = asyncio.create_task(_remember_later(state["question"]))
        _background.add(task)
        task.add_done_callback(_background.discard)
    return {}


def _after_route(state: StudyState) -> str:
    return {"connect": "connect", "quiz": "quiz_start", "review": "quiz_start", "quiz_answer": "quiz_answer",
            "quiz_stop": "quiz_stop"}.get(state["intent"], "retrieve")


def _after_grade(state: StudyState) -> str:
    if state["documents"]:
        return "generate"
    # Nothing indexed for this scope: a rewrite cannot find chunks that don't exist
    if state["found"] == 0 or state["retries"] >= MAX_RETRIES:
        return "not_found"
    return "rewrite_query"


def _after_rewrite(state: StudyState) -> str:
    # Nothing new to search for: the same weak search would just run again
    return "not_found" if state["query"] in state["queries_tried"] else "retrieve"


def _after_check(state: StudyState) -> str:
    return "generate" if state["unsupported"] and not state["regenerated"] else "respond"


def build(checkpointer):
    g = StateGraph(StudyState)
    for node in (route_intent, retrieve, grade_documents, rewrite_query, generate, self_check,
                 not_found, connect, quiz_start, quiz_answer, quiz_stop, respond, remember):
        g.add_node(node.__name__, node)
    g.add_edge(START, "route_intent")
    g.add_conditional_edges("route_intent", _after_route,
                            ["retrieve", "connect", "quiz_start", "quiz_answer", "quiz_stop"])
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", _after_grade, ["generate", "rewrite_query", "not_found"])
    g.add_conditional_edges("rewrite_query", _after_rewrite, ["retrieve", "not_found"])
    g.add_edge("generate", "self_check")
    g.add_conditional_edges("self_check", _after_check, ["generate", "respond"])
    for node in ("not_found", "connect", "quiz_start", "quiz_answer", "quiz_stop"):
        g.add_edge(node, "respond")
    g.add_edge("respond", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph(path: Optional[str] = None):
    """The compiled agent, with conversations checkpointed in SQLite by thread_id."""
    async with AsyncSqliteSaver.from_conn_string(path or settings.agent_db_path) as saver:
        yield build(saver)
