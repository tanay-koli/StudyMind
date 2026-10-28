"""
StudyMind — Query Module

Every feature retrieves with a short query (the question or topic), then
generates with the local LLM over the numbered passages; never one string for both.
  1. ask()              — Q&A over all material, with citations
  2. ask_scoped()       — the same, filtered to one topic
  3. cross_topic_ask()  — primary topic answer + connections from other topics
  4. connect_topics()   — shared concepts between two topics, as a graph
  5. summarize_topic()  — cross-source summary of a free-text topic
"""

import asyncio
import hashlib
import re
from typing import Optional

from core.config import settings
from core.graph_utils import render_pyvis_graph, SOURCE_COLORS
from core.llm import llm
from core.retrieval import Hit, get_retriever

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

    Retrieves the question per topic (top 6 primary, top 2 each other topic),
    then generates once over those passages only.

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
        others = [tid for tid in all_topic_ids if tid != primary_topic_id]
        primary_hits, *other_lists = await asyncio.gather(
            retriever.retrieve(question, primary_topic_id, k=6),
            *(retriever.retrieve(question, tid, k=2) for tid in others),
        )
        other_hits = [h for hits in other_lists for h in hits]
        if not primary_hits and not other_hits:
            result["answer"] = (
                f"No results found in **{primary_name}** or any other topic. "
                "Try adding more material or asking a broader question."
            )
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
    """Find and visualize the concepts connecting two topics.

    Retrieves each topic's passages; when both topics have material, the LLM
    names the concepts connecting them. Otherwise a clean empty state, so
    nothing is invented for a topic the material does not cover.

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

    try:
        retriever = get_retriever()
        hits_a, hits_b = await asyncio.gather(
            retriever.retrieve(topic_a, None, k=6),
            retriever.retrieve(topic_b, None, k=6),
        )
        if hits_a and hits_b:
            nodes, edges, summary = await _connect_via_llm(topic_a, topic_b, hits_a, hits_b)
            if nodes and edges:
                result.update(
                    nodes=nodes, edges=edges, summary=summary, connection_method="ai_synthesized",
                    html_graph=render_pyvis_graph(nodes, edges, topic_a, topic_b),
                )
                return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["summary"] = (
        f"No connections found between '{topic_a}' and '{topic_b}' "
        f"in your material. Try ingesting more material that covers both topics."
    )
    return result


async def _connect_via_llm(topic_a: str, topic_b: str, hits_a: list[Hit], hits_b: list[Hit]) -> tuple:
    """The LLM names connecting concepts from both topics' passages."""
    # A chunk retrieved for both topics goes in once
    seen = {h.chunk_id for h in hits_a}
    hits_b = [h for h in hits_b if h.chunk_id not in seen]
    ctx_a, _ = _context(hits_a)
    ctx_b, _ = _context(hits_b, start=len(hits_a) + 1)
    reply = await llm.complete(
        f"Passages about '{topic_a}':\n{ctx_a or '(none)'}\n\n"
        f"Passages about '{topic_b}':\n{ctx_b or '(none)'}\n\n"
        f"If the passages for either topic are about a completely different subject, "
        f"reply exactly NONE.\n"
        f"Otherwise list up to 4 concepts from these passages that connect '{topic_a}' and "
        f"'{topic_b}', one per line, exactly as:\nCONCEPT: <name> | REASON: <one sentence>",
        system="You find connections between study topics using only the given passages.",
    )
    if not reply.strip():
        return [], [], ""

    nodes, edges = _parse_concept_lines(reply, topic_a, topic_b)
    if len(nodes) <= 2:
        return [], [], ""

    concept_nodes = [n for n in nodes if n.get("source_type") == "shared"]
    summary = (
        f"**Connections suggested by the local LLM** (from passages on both topics):\n\n"
        + "\n".join(f"- **{n['label']}** — {n.get('title', '')}" for n in concept_nodes)
    )
    return nodes, edges, summary


def _parse_concept_lines(text: str, topic_a: str, topic_b: str) -> tuple:
    """Parse an LLM reply into pyvis nodes + edges.

    Expected line format (one per concept):
        CONCEPT: <name> | REASON: <one sentence>

    Falls back gracefully to any bullet/numbered line.
    """
    nodes: list = []
    edges: list = []
    seen_ids: set = set()

    def _add_node(label: str, stype: str, title: str = "") -> str:
        nid = _node_id(label)
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label[:50],
                "color": SOURCE_COLORS.get(stype, SOURCE_COLORS["default"]),
                "source_type": stype,
                "title": title or label,
            })
        return nid

    id_a = _add_node(topic_a, "topic", f"Topic: {topic_a}")
    id_b = _add_node(topic_b, "topic", f"Topic: {topic_b}")

    concepts: list = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        concept_name = ""
        reason = ""
        # Small models number, bullet and bold the lines: "1. **CONCEPT:** X | **REASON:** y"
        plain = re.sub(r"^[-*•\d.)\s]+", "", line.replace("*", "")).strip()

        m = re.match(r"(?i)concept:\s*(.+?)\s*(?:\||\s[–—-]\s)\s*(?:reason:\s*)?(.*)$", plain)
        if m:
            concept_name, reason = m.group(1).strip(), m.group(2).strip()
        elif "|" in plain:
            left, right = (p.strip() for p in plain.split("|", 1))
            concept_name = re.sub(r"(?i)^concept:\s*", "", left)
            reason = re.sub(r"(?i)^reason:\s*", "", right)
        else:
            m = re.match(r"^[-*•\d]+[.):]?\s*(.+)$", line)
            if m:
                concept_name = re.sub(r"(?i)^concept:\s*", "", m.group(1).replace("*", "").strip())

        if not concept_name:
            continue
        if concept_name.lower() in (topic_a.lower(), topic_b.lower()):
            continue
        if len(concept_name) > 120:
            concept_name = concept_name[:120]

        concepts.append((concept_name, reason))
        if len(concepts) >= 4:
            break

    for concept_name, reason in concepts:
        title = f"{concept_name}: {reason}" if reason else concept_name
        cid = _add_node(concept_name, "shared", title)
        edges.append({
            "from_id": id_a,
            "to_id": cid,
            "from_label": topic_a,
            "to_label": concept_name,
            "relationship": "relates to",
        })
        edges.append({
            "from_id": cid,
            "to_id": id_b,
            "from_label": concept_name,
            "to_label": topic_b,
            "relationship": "relates to",
        })

    return nodes, edges
