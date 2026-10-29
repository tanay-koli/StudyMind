"""
StudyMind — Query Module

Every feature retrieves with a short query (the question or topic), then
generates with the local LLM over the numbered passages; never one string for both.
  1. ask()              — Q&A over all material, with citations
  2. ask_scoped()       — the same, filtered to one topic
  3. cross_topic_ask()  — primary topic answer + connections from other topics
  4. connect_topics()   — concepts shared by two topics (stored triples), as a graph
  5. summarize_topic()  — cross-source summary of a free-text topic
"""

import asyncio
import hashlib
import re
from itertools import zip_longest
from typing import Optional

from sqlmodel import select

from core import concepts
from core.config import settings
from core.db import Chunk, session
from core.graph_utils import render_pyvis_graph, SOURCE_COLORS
from core.llm import llm
from core.retrieval import Hit, get_retriever, relevant

CITE_RULES = (
    "You are a study assistant. Answer only from the numbered context passages.\n"
    "Cite the passage for every claim with its number in square brackets, like [2].\n"
)
ANSWER_SYSTEM = CITE_RULES + "If the context does not contain the answer, reply exactly: This isn't in your material."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_id(label: str) -> str:
    """Stable 8-char ID from a label string."""
    return hashlib.md5(label.lower().encode()).hexdigest()[:8]


def _context(hits: list[Hit], start: int = 1) -> tuple[str, list[dict]]:
    """Numbered passages for the prompt, and the matching citations."""
    text = "\n\n".join(
        f"[{n}] {h.source_title}"
        + (f", p.{h.page}" if h.page else "")
        + (f" — {h.section}" if h.section else "")
        + f"\n{h.text}"
        for n, h in enumerate(hits, start=start)
    )
    citations = [
        {"source_title": h.source_title, "page": h.page, "chunk_id": h.chunk_id,
         "score": h.score, "text": h.text}
        for h in hits
    ]
    return text, citations


def _topic_name(topic_id: str) -> str:
    from core.topics import load_topics

    return load_topics().get(topic_id, {}).get("name", topic_id)


def _topics_with_chunks(topic_ids: list[str]) -> set[str]:
    with session() as s:
        return set(s.exec(select(Chunk.topic_id).where(Chunk.topic_id.in_(topic_ids)).distinct()).all())


async def _sub_questions(question: str) -> list[str]:
    """Up to 3 sub-questions the LLM splits a cross-topic question into; [question] if it can't."""
    try:
        reply = await llm.complete(
            f"Split this question into 2 or 3 short, self-contained sub-questions, one per line, "
            f"each ending in '?'. Output only the sub-questions.\n\nQuestion: {question}"
        )
    except Exception:
        return [question]
    lines = (re.sub(r"^[-*•\d.)\s]+", "", ln.replace("*", "")).strip() for ln in reply.splitlines())
    subs = list(dict.fromkeys(ln for ln in lines if ln.endswith("?")))[:3]
    return subs if len(subs) >= 2 else [question]


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------

async def _answer(question: str, topic_id: Optional[str], empty_msg: str) -> dict:
    """Retrieve chunks, then generate a grounded answer that cites them as [n]."""
    result = {
        "answer": "",
        "sources": [],
        "citations": [],
        "search_type": settings.retriever.upper(),
        "error": None,
    }
    try:
        hits = await get_retriever().retrieve(question, topic_id)
        if not hits:
            result["answer"] = empty_msg
            return result
        context, result["citations"] = _context(hits)
        result["sources"] = list(dict.fromkeys(h.source_title for h in hits))
        result["answer"] = await llm.complete(f"{context}\n\nQuestion: {question}", system=ANSWER_SYSTEM)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


async def ask(question: str) -> dict:
    """Answer from all ingested material.

    Returns:
        dict with keys: answer, sources, citations, search_type, error
    """
    return await _answer(
        question, None,
        "No results found. Try ingesting more material or rephrasing your question.",
    )


async def ask_scoped(question: str, topic_id: str) -> dict:
    """Answer from one topic's material only.

    Returns:
        dict with keys: answer, sources, citations, search_type, topic_id, error
    """
    result = await _answer(
        question, topic_id,
        f"No results found in **{_topic_name(topic_id)}**. "
        "Try adding more material to this topic or asking a broader question.",
    )
    result["topic_id"] = topic_id
    return result


async def cross_topic_ask(question: str, primary_topic_id: str, all_topic_ids: list) -> dict:
    """Answer from the primary topic, plus connections from every other topic.

    Splits the question into sub-questions, retrieves each per topic (top 6
    primary, top 2 each other topic), then generates once over those passages only.

    Returns:
        dict with keys: answer, primary_section, cross_section, sources, citations,
                        search_type, topic_id, error
    """
    primary_name = _topic_name(primary_topic_id)
    result = {
        "answer": "",
        "primary_section": "",
        "cross_section": "",
        "sources": [],
        "citations": [],
        "search_type": f"{settings.retriever.upper()} (cross-topic)",
        "topic_id": primary_topic_id,
        "error": None,
    }
    try:
        retriever = get_retriever()
        present = _topics_with_chunks(list(all_topic_ids))
        no_results = (
            f"No results found in **{primary_name}** or any other topic. "
            "Try adding more material or asking a broader question."
        )
        if not present:
            result["answer"] = no_results
            return result
        subs = await _sub_questions(question)
        others = [tid for tid in all_topic_ids if tid != primary_topic_id and tid in present]

        async def per_topic(tid: str, k: int) -> list[Hit]:
            lists = await asyncio.gather(*(retriever.retrieve(q, tid, k=k) for q in subs))
            # Round-robin so a later sub-question isn't crowded out by the first one's k hits
            merged = {h.chunk_id: h for row in zip_longest(*lists) for h in row if h}
            return list(merged.values())[:k]

        primary_hits, *other_lists = await asyncio.gather(
            per_topic(primary_topic_id, 6), *(per_topic(tid, 2) for tid in others)
        )
        other_hits = [h for hits in other_lists for h in hits]
        if not primary_hits and not other_hits:
            result["answer"] = no_results
            return result

        ctx_primary, cites_primary = _context(primary_hits)
        ctx_other, cites_other = _context(other_hits, start=len(primary_hits) + 1)
        result["citations"] = cites_primary + cites_other
        result["sources"] = list(dict.fromkeys(h.source_title for h in primary_hits + other_hits))
        prompt = (
            f"Passages from {primary_name}:\n{ctx_primary or '(none)'}\n\n"
            f"Passages from other topics:\n{ctx_other or '(none)'}\n\n"
            f"Question: {question}\n\n"
            f"Answer in two sections:\n"
            f"**From {primary_name}:** the answer from its passages\n\n"
            f"**Connected from other topics:** related points from the other passages, or 'None found'"
        )
        answer = await llm.complete(prompt, system=CITE_RULES)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["answer"] = answer
    # Tolerate "**X:**", "**X**:", "### X" and plain "X:" headings
    parts = re.split(r"(?im)^[#*\s]*connected from other topics[*:\s]*", answer, maxsplit=1)
    head = rf"(?i)^[#*\s]*from {re.escape(primary_name)}[*:\s]*"
    result["primary_section"] = re.sub(head, "", parts[0].strip()).strip()
    result["cross_section"] = parts[1].strip() if len(parts) > 1 else ""
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

async def summarize_topic(topic: str) -> dict:
    """Summarize what the material says about a free-text topic.

    Returns:
        dict with keys: summary, sources_referenced, citations, error
    """
    result: dict = {"summary": "", "sources_referenced": [], "citations": [], "error": None}
    try:
        hits = await get_retriever().retrieve(topic, None, k=8)
        if not hits:
            result["summary"] = (
                f"No information found about '{topic}'. "
                f"Make sure you've ingested material that covers this topic."
            )
            return result
        context, result["citations"] = _context(hits)
        result["sources_referenced"] = list(dict.fromkeys(h.source_title for h in hits))
        result["summary"] = await llm.complete(
            f"{context}\n\nSummarize what these passages say about '{topic}'.",
            system=CITE_RULES,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

async def connect_topics(topic_a: str, topic_b: str) -> dict:
    """Find and visualize the concepts shared by two topics' retrieved passages.

    Overlap of the stored per-chunk concept triples; no LLM call, so nothing is
    invented for a topic the material does not cover.

    Returns:
        dict with keys: nodes, edges, summary, html_graph, connection_method, error
    """
    result: dict = {
        "nodes": [],
        "edges": [],
        "summary": "",
        "html_graph": None,
        "connection_method": "none",
        "error": None,
    }
    pending = ""
    try:
        retriever = get_retriever(rewrite=False)  # spec: connect makes no LLM call
        hits_a, hits_b = await asyncio.gather(
            retriever.retrieve(topic_a, None, k=concepts.CONNECT_K),
            retriever.retrieve(topic_b, None, k=concepts.CONNECT_K),
        )
        # Without a cutoff an unrelated topic still returns its k nearest chunks, and overlaps get invented
        hits_a, hits_b = relevant(hits_a), relevant(hits_b)
        ids = list(dict.fromkeys(h.chunk_id for h in hits_a + hits_b))
        if (n := concepts.pending_count(ids)):
            pending = "\n\n" + concepts.PENDING_NOTE.format(n=n, m=len(ids))
        shared = concepts.shared_concepts(
            [h.chunk_id for h in hits_a], [h.chunk_id for h in hits_b],
            {concepts.normalise(topic_a), concepts.normalise(topic_b)},
        )
        if shared:
            by_chunk = {h.chunk_id: h for h in hits_a + hits_b}

            def cite(e) -> str:
                h = by_chunk[e.chunk_id]
                return f"{e.head} {e.relation} {e.tail} (📄 {h.source_title}" + (f" · p.{h.page}" if h.page else "") + ")"

            def node(label: str, stype: str, title: str) -> dict:
                return {"id": _node_id(label), "label": label[:50], "color": SOURCE_COLORS.get(stype, SOURCE_COLORS["default"]),
                        "source_type": stype, "title": title}

            nodes = [node(topic_a, "topic", f"Topic: {topic_a}"), node(topic_b, "topic", f"Topic: {topic_b}")]
            edges, lines = [], []
            for sh in shared:
                nodes.append(node(sh.name, "shared", sh.name))
                edges.append({"from_id": _node_id(topic_a), "to_id": _node_id(sh.name), "from_label": topic_a,
                              "to_label": sh.name, "relationship": sh.edge_a.relation})
                edges.append({"from_id": _node_id(sh.name), "to_id": _node_id(topic_b), "from_label": sh.name,
                              "to_label": topic_b, "relationship": sh.edge_b.relation})
                lines.append(f"- **{sh.name}** — A: {cite(sh.edge_a)}; B: {cite(sh.edge_b)}")
            result.update(
                nodes=nodes, edges=edges, summary="\n".join(lines) + pending, connection_method="concept_overlap",
                html_graph=render_pyvis_graph(nodes, edges, topic_a, topic_b),
            )
            return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["summary"] = (
        f"No connections found between '{topic_a}' and '{topic_b}' "
        f"in your material. Try ingesting more material that covers both topics."
    ) + pending
    return result
