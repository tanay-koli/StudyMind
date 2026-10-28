"""
StudyMind — Query Module

Five core query functions:
  1. ask()              — Q&A over all material: our retriever + local LLM, with citations
  2. ask_scoped()       — the same, filtered to one topic
  3. cross_topic_ask()  — primary topic Q&A + cross-topic connections
  4. connect_topics()   — relationship discovery (two CHUNKS recalls + synthesis)
  5. summarize_topic()  — cross-source summary (HYBRID_COMPLETION)

All functions are async and share one httpx.AsyncClient (core.config.http_client).

Search types supported by Cognee Cloud /api/v1/recall:
  HYBRID_COMPLETION     — vector + graph blend (fast, default for summaries)
  GRAPH_COMPLETION      — full graph reasoning (best for Q&A)
  CHUNKS                — pure vector retrieval, no LLM (for connect_topics context)
  GRAPH_SUMMARY_COMPLETION — graph + summary LLM

datasets filter (confirmed from API spec):
  Pass datasets=[topic_id] to scope recall to a single topic's data.
"""

import asyncio
import hashlib
import re
from typing import Optional

import httpx

from core.config import settings, _headers, http_client
from core.llm import llm
from core.retrieval import get_retriever
from core.graph_utils import (
    parse_search_results,
    build_connection_graph,
    format_source_attribution,
    render_pyvis_graph,
    SOURCE_COLORS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_id(label: str) -> str:
    """Stable 8-char ID from a label string."""
    return hashlib.md5(label.lower().encode()).hexdigest()[:8]


def _chunk_text(text: str, max_chars: int) -> str:
    """Truncate a chunk to max_chars with a trailing ellipsis."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


async def _recall(
    query: str,
    search_type: str = "GRAPH_COMPLETION",
    datasets: Optional[list] = None,
    timeout: int = 60,
) -> dict:
    """POST /api/v1/recall.

    Args:
        query: Natural language query string.
        search_type: One of GRAPH_COMPLETION, HYBRID_COMPLETION, CHUNKS,
                     GRAPH_SUMMARY_COMPLETION.
        datasets: Optional list of dataset names to scope the search
                  (maps to the 'datasets' field in the Cognee Cloud API).
        timeout: HTTP timeout in seconds.

    Returns:
        dict with keys: ok (bool), data (any), error (str|None)
    """
    if not settings.cognee_base_url or not settings.cognee_api_key:
        return {
            "ok": False,
            "data": None,
            "error": "Cognee Cloud credentials not configured. Check .env.",
        }

    payload: dict = {"query": query, "search_type": search_type}
    if datasets:
        payload["datasets"] = datasets

    try:
        resp = await http_client().post(
            f"{settings.cognee_base_url}/api/v1/recall",
            headers=_headers(),
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return {"ok": True, "data": resp.json(), "error": None}
        else:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:300]
            return {
                "ok": False,
                "data": None,
                "error": f"Cognee Cloud returned HTTP {resp.status_code}: {detail}",
            }
    except httpx.ConnectError:
        return {
            "ok": False,
            "data": None,
            "error": (
                f"Cannot reach Cognee Cloud at {settings.cognee_base_url}. "
                "Check your internet connection."
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": f"Recall failed: {type(exc).__name__}: {exc}",
        }


def _extract_text_from_recall(data) -> list[str]:
    """Pull answer/text strings out of the Cognee Cloud recall response.

    The cloud API may return:
      - A list of result objects  [{text: ...}, ...]
      - A string answer directly
      - A dict with an "answer" or "results" key
    """
    texts = []

    if data is None:
        return texts

    if isinstance(data, str):
        return [data] if data.strip() else []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                if item.strip():
                    texts.append(item)
            elif isinstance(item, dict):
                t = (
                    item.get("text", "")
                    or item.get("answer", "")
                    or item.get("content", "")
                    or item.get("name", "")
                    or str(item)
                )
                if t.strip():
                    texts.append(t.strip())
        return texts

    if isinstance(data, dict):
        # Try common response shapes
        if "answer" in data:
            return [data["answer"]] if data["answer"] else []
        if "results" in data:
            return _extract_text_from_recall(data["results"])
        if "text" in data:
            return [data["text"]] if data["text"] else []
        # Fallback: stringify
        return [str(data)]

    return [str(data)]


def _extract_sources_from_recall(data) -> list[str]:
    """Extract source/dataset attribution strings from recall response."""
    sources = []
    seen: set = set()

    items = data if isinstance(data, list) else ([data] if data else [])
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("dataset_name", "source", "dataset"):
            val = item.get(key)
            if val and isinstance(val, str) and val not in seen:
                sources.append(val)
                seen.add(val)

    return sources


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are a study assistant. Answer only from the numbered context passages.\n"
    "Cite the passage for every claim with its number in square brackets, like [2].\n"
    "If the context does not contain the answer, reply exactly: This isn't in your material."
)


async def _answer(question: str, topic_id: Optional[str], empty_msg: str) -> dict:
    """Retrieve chunks, then generate a grounded answer that cites them as [n]."""
    retriever = get_retriever()
    result = {
        "answer": "",
        "sources": [],
        "citations": [],
        "search_type": "COGNEE_CHUNKS" if settings.retriever == "cognee" else "HYBRID",
        "error": None,
    }
    try:
        hits = await retriever.retrieve(question, topic_id)
        if not hits:
            result["answer"] = empty_msg
            return result
        context = "\n\n".join(
            f"[{n}] {h.source_title}"
            + (f", p.{h.page}" if h.page else "")
            + (f" — {h.section}" if h.section else "")
            + f"\n{h.text}"
            for n, h in enumerate(hits, start=1)
        )
        result["citations"] = [
            {"source_title": h.source_title, "page": h.page, "chunk_id": h.chunk_id,
             "score": h.score, "text": h.text}
            for h in hits
        ]
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
    from core.topics import load_topics

    topic_name = load_topics().get(topic_id, {}).get("name", topic_id)
    result = await _answer(
        question, topic_id,
        f"No results found in **{topic_name}**. "
        "Try adding more material to this topic or asking a broader question.",
    )
    result["topic_id"] = topic_id
    return result


async def cross_topic_ask(
    question: str,
    primary_topic_id: str,
    all_topic_ids: list,
) -> dict:
    """Answer using primary topic + surface related content from other topics.

    Strategy:
      1. Recall CHUNKS from primary topic (top 4 chunks, 300 chars each)
      2. Recall CHUNKS from every other topic (top 2 chunks each)
      3. Synthesize with ONE GRAPH_COMPLETION recall using a structured prompt
         that separates primary vs cross-topic context

    Returns:
        dict with keys: answer, primary_section, cross_section, sources,
                        search_type, topic_id, error
    """
    from core.topics import get_topic_datasets, load_topics

    result = {
        "answer": "",
        "primary_section": "",
        "cross_section": "",
        "sources": [],
        "search_type": "GRAPH_COMPLETION (cross-topic)",
        "topic_id": primary_topic_id,
        "error": None,
    }

    topics = load_topics()
    primary_name = topics.get(primary_topic_id, {}).get("name", primary_topic_id)
    primary_datasets = get_topic_datasets(primary_topic_id)

    # Step 1: fetch primary topic chunks
    primary_recall = await _recall(
        question,
        search_type="CHUNKS",
        datasets=primary_datasets if primary_datasets else None,
        timeout=30,
    )
    primary_texts = _extract_text_from_recall(primary_recall.get("data")) if primary_recall["ok"] else []
    primary_chunks = [t[:300] for t in primary_texts[:4]]

    # Step 2: fetch chunks from other topics
    other_chunks: list = []
    other_ids = [
        tid for tid in all_topic_ids
        if tid != primary_topic_id and get_topic_datasets(tid)
    ]
    other_recalls = await asyncio.gather(*(
        _recall(question, search_type="CHUNKS", datasets=get_topic_datasets(tid), timeout=20)
        for tid in other_ids
    ))
    for other_id, other_recall in zip(other_ids, other_recalls):
        other_name = topics.get(other_id, {}).get("name", other_id)
        if other_recall["ok"] and other_recall["data"]:
            texts = _extract_text_from_recall(other_recall["data"])
            for t in texts[:2]:
                other_chunks.append(f"[{other_name}] {t[:300]}")

    # Step 3: synthesize
    ctx_primary = "\n".join(f"- {c}" for c in primary_chunks) if primary_chunks else f"(no material found in '{primary_name}')"
    ctx_cross   = "\n".join(f"- {c}" for c in other_chunks) if other_chunks else "(no related content found in other topics)"

    synthesis_prompt = (
        f"Answer '{question}' primarily using the '{primary_name}' material, "
        f"but also note any relevant connections from other topics.\n\n"
        f"Primary source ({primary_name}):\n{ctx_primary}\n\n"
        f"Related from other topics:\n{ctx_cross}\n\n"
        f"Structure your answer in two sections:\n"
        f"**From {primary_name}:** [answer from primary material]\n\n"
        f"**Connected from other topics:** [any relevant connections, or 'None found' if none]"
    )

    synth_recall = await _recall(synthesis_prompt, search_type="GRAPH_COMPLETION", timeout=90)

    if synth_recall["ok"] and synth_recall["data"]:
        full_answer = "\n\n".join(_extract_text_from_recall(synth_recall["data"]))
        result["answer"] = full_answer
        result["sources"] = _extract_sources_from_recall(synth_recall["data"])

        # Try to split into sections for UI display
        if f"**From {primary_name}:**" in full_answer:
            parts = full_answer.split(f"**Connected from other topics:**", 1)
            result["primary_section"] = parts[0].replace(f"**From {primary_name}:**", "").strip()
            result["cross_section"]   = parts[1].strip() if len(parts) > 1 else ""
        else:
            result["primary_section"] = full_answer
            result["cross_section"]   = ""
    else:
        result["error"] = synth_recall.get("error")
        result["answer"] = (
            f"**From {primary_name}:**\n" + "\n".join(f"- {c}" for c in primary_chunks)
            + (f"\n\n**Connected from other topics:**\n" + "\n".join(f"- {c}" for c in other_chunks) if other_chunks else "")
        )

    return result


async def connect_topics(topic_a: str, topic_b: str) -> dict:

    """Find and visualize the relationship path between two topics.

    Strategy (Cognee Cloud):
      Option A: Recall CHUNKS for each topic → find overlapping entities →
                build pyvis graph from shared concepts.
      Option B: Single GRAPH_COMPLETION recall combining both topics →
                parse plain-text concepts → build pyvis graph.
      Option C: Clean empty state — no misleading fabricated connections.

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

    # Option A: CHUNKS retrieval → entity overlap
    try:
        nodes, edges, summary = await _connect_via_chunks(topic_a, topic_b)
        if nodes and edges:
            result["nodes"] = nodes
            result["edges"] = edges
            result["summary"] = summary
            result["connection_method"] = "entity_overlap"
            result["html_graph"] = render_pyvis_graph(nodes, edges, topic_a, topic_b)
            return result
    except Exception as exc_a:
        result["error"] = f"[Option A] {type(exc_a).__name__}: {exc_a}"

    # Option B: GRAPH_COMPLETION for the combined query
    try:
        nodes, edges, summary = await _connect_via_graph_completion(topic_a, topic_b)
        if nodes and edges:
            result["nodes"] = nodes
            result["edges"] = edges
            result["summary"] = summary
            result["connection_method"] = "ai_synthesized"
            result["html_graph"] = render_pyvis_graph(nodes, edges, topic_a, topic_b)
            return result
    except Exception as exc_b:
        prev = result.get("error") or ""
        result["error"] = prev + f" | [Option B] {type(exc_b).__name__}: {exc_b}"

    # Option C: clean empty state
    result["summary"] = (
        f"No connections found between '{topic_a}' and '{topic_b}' "
        f"in the current knowledge graph. Try ingesting more material "
        f"that covers both topics."
    )
    result["connection_method"] = "none"
    result["html_graph"] = None
    return result


async def summarize_topic(topic: str) -> dict:
    """Generate a cross-source summary for a given topic.

    Uses HYBRID_COMPLETION for a balanced speed/quality tradeoff.

    Returns:
        dict with keys: summary (str), sources_referenced (List[str]), error (str|None)
    """
    result: dict = {
        "summary": "",
        "sources_referenced": [],
        "error": None,
    }

    recall = await _recall(topic, search_type="HYBRID_COMPLETION")
    if not recall["ok"]:
        # Fallback to CHUNKS
        recall = await _recall(topic, search_type="CHUNKS")

    if not recall["ok"]:
        result["error"] = recall.get("error")
        result["summary"] = (
            f"Could not retrieve information about '{topic}'. "
            f"Error: {result['error']}"
        )
        return result

    data = recall["data"]
    texts = _extract_text_from_recall(data)
    sources = _extract_sources_from_recall(data)

    if not texts:
        result["summary"] = (
            f"No information found about '{topic}'. "
            f"Make sure you've ingested material that covers this topic."
        )
        return result

    result["summary"] = "\n\n".join(texts)
    result["sources_referenced"] = sources
    return result


# ---------------------------------------------------------------------------
# Option A internals — CHUNKS overlap
# ---------------------------------------------------------------------------

async def _connect_via_chunks(topic_a: str, topic_b: str) -> tuple:
    """Retrieve CHUNKS for each topic and find overlapping entities.

    Returns (nodes, edges, summary). Returns ([], [], '') if no overlap.
    """
    raw_a, raw_b = await asyncio.gather(
        _recall(topic_a, search_type="CHUNKS", timeout=30),
        _recall(topic_b, search_type="CHUNKS", timeout=30),
    )

    texts_a = _extract_text_from_recall(raw_a.get("data")) if raw_a["ok"] else []
    texts_b = _extract_text_from_recall(raw_b.get("data")) if raw_b["ok"] else []

    # Convert text lists into the normalized dict format expected by build_connection_graph
    parsed_a = [{"text": t, "type": "chunk", "source": None, "metadata": {}, "raw": None} for t in texts_a]
    parsed_b = [{"text": t, "type": "chunk", "source": None, "metadata": {}, "raw": None} for t in texts_b]

    nodes, edges = build_connection_graph(topic_a, topic_b, parsed_a, parsed_b)

    if not nodes or not edges:
        return [], [], ""

    concept_labels = [
        n["label"] for n in nodes
        if n.get("source_type") == "shared"
    ]
    summary = (
        f"**Shared concepts connecting '{topic_a}' and '{topic_b}'** "
        f"(found via knowledge graph retrieval):\n\n"
        + "\n".join(f"- **{label}**" for label in concept_labels)
    )
    return nodes, edges, summary


# ---------------------------------------------------------------------------
# Option B internals — GRAPH_COMPLETION synthesis
# ---------------------------------------------------------------------------

def _parse_concept_lines(text: str, topic_a: str, topic_b: str) -> tuple:
    """Parse plain-text recall response into pyvis nodes + edges.

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

        if "|" in line:
            parts = line.split("|", 1)
            left  = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""

            for prefix in ("CONCEPT:", "concept:", "Concept:"):
                if left.startswith(prefix):
                    left = left[len(prefix):].strip()
                    break
            for prefix in ("REASON:", "reason:", "Reason:"):
                if right.startswith(prefix):
                    right = right[len(prefix):].strip()
                    break

            concept_name = left
            reason = right
        else:
            m = re.match(r"^[-*\u2022\d]+[.):]?\s*(.+)$", line)
            if m:
                concept_name = m.group(1).strip()

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


async def _connect_via_graph_completion(topic_a: str, topic_b: str) -> tuple:
    """Option B: single GRAPH_COMPLETION recall combining both topics.

    Returns (nodes, edges, summary_text). Returns ([], [], '') if empty output.
    """
    combined_query = (
        f"How are '{topic_a}' and '{topic_b}' related? "
        f"List the key concepts that connect them."
    )

    recall = await _recall(combined_query, search_type="GRAPH_COMPLETION", timeout=60)
    if not recall["ok"]:
        return [], [], ""

    texts = _extract_text_from_recall(recall["data"])
    raw_text = "\n".join(texts)

    if not raw_text.strip():
        return [], [], ""

    nodes, edges = _parse_concept_lines(raw_text, topic_a, topic_b)

    if not nodes or len(nodes) <= 2:
        return [], [], ""

    concept_nodes = [n for n in nodes if n.get("source_type") == "shared"]
    summary = (
        f"**Connections synthesized by Cognee Cloud** "
        f"(graph reasoning across your study material):\n\n"
        + "\n".join(
            f"- **{n['label']}** — {n.get('title', '')}" for n in concept_nodes
        )
    )

    return nodes, edges, summary


# ---------------------------------------------------------------------------
# _build_text_summary() — kept for backward compatibility
# ---------------------------------------------------------------------------

def _build_text_summary(topic_a: str, topic_b: str, nodes: list, edges: list) -> str:
    """Build a simple text summary from graph data."""
    if not edges:
        return f"No connections found between '{topic_a}' and '{topic_b}'."

    lines = [f"**Connection path between '{topic_a}' and '{topic_b}':**\n"]
    for edge in edges:
        lines.append(
            f"- **{edge.get('from_label', '?')}** --[{edge.get('relationship', 'related')}]--> "
            f"**{edge.get('to_label', '?')}**"
        )
    return "\n".join(lines)
