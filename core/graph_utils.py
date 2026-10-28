"""
StudyMind — Graph Utilities

Helpers to build connection graphs from overlapping passages and render
interactive pyvis visualizations.
"""

import hashlib
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Color palette for source types
# ---------------------------------------------------------------------------
SOURCE_COLORS = {
    "pdf": "#FF6B6B",       # Coral red
    "youtube": "#FF4757",   # YouTube red
    "web": "#2ED573",       # Fresh green
    "note": "#FFA502",      # Warm amber
    "topic": "#7C5CFC",     # Primary purple
    "shared": "#1E90FF",    # Connecting blue
    "default": "#A4B0BE",   # Neutral gray
}


def _extract_entity_names(parsed_results: list[dict]) -> set[str]:
    """Extract entity/concept names from parsed results.

    Looks for capitalized multi-word terms, quoted terms, and
    structured entity references in the results.
    """
    entities = set()
    for item in parsed_results:
        text = item.get("text", "")
        if not text:
            continue

        # Add the raw text as a potential entity (for graph nodes)
        # Truncate long texts to a reasonable label
        clean = text.strip()
        if len(clean) <= 80:
            entities.add(clean)
        else:
            # For longer texts, try to extract key terms
            # Look for text that appears structured (from graph context)
            lines = clean.split("\n")
            for line in lines[:10]:
                line = line.strip()
                if line and len(line) <= 80:
                    entities.add(line)

        # Also check metadata for structured entities
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("name", "subject", "object", "predicate", "entity"):
                val = metadata.get(key)
                if val and isinstance(val, str):
                    entities.add(val)

    return entities


def build_connection_graph(
    topic_a: str,
    topic_b: str,
    results_a: list[dict],
    results_b: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Find overlapping entities between two topic searches and build a graph.

    NOTE: This function is kept for backward compatibility with code that still
    calls it directly. The preferred path is query.py's _connect_via_graph()
    (Option A) or _connect_via_llm() (Option B), both of which produce real
    or honestly-labelled connections.

    Strategy:
    1. Extract entity names from both result sets
    2. Find entities that appear in both sets (overlap = connection points)
    3. Build a node-edge graph: topic_a -> shared entities -> topic_b

    Fix (2026-07-01): Removed the misleading '?' bridge node. When no overlap
    is found, return ([], []) instead of fabricating a fake connection.

    Args:
        topic_a: First topic name.
        topic_b: Second topic name.
        results_a: Parsed results from searching topic_a.
        results_b: Parsed results from searching topic_b.

    Returns:
        Tuple of (nodes, edges). Returns ([], []) when no real overlap exists.
    """
    nodes = []
    edges = []
    node_ids = set()

    def _node_id(label: str) -> str:
        return hashlib.md5(label.lower().encode()).hexdigest()[:8]

    def _add_node(label: str, source_type: str = "default", title: str = "") -> str:
        nid = _node_id(label)
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label[:50],
                "color": SOURCE_COLORS.get(source_type, SOURCE_COLORS["default"]),
                "source_type": source_type,
                "title": title or label,
            })
        return nid

    # Add the two main topic nodes
    id_a = _add_node(topic_a, "topic", f"Topic: {topic_a}")
    id_b = _add_node(topic_b, "topic", f"Topic: {topic_b}")

    # Extract entities from both sets
    entities_a = _extract_entity_names(results_a)
    entities_b = _extract_entity_names(results_b)

    # Find shared entities (case-insensitive overlap)
    entities_a_lower = {e.lower(): e for e in entities_a}
    entities_b_lower = {e.lower(): e for e in entities_b}

    shared_keys = set(entities_a_lower.keys()) & set(entities_b_lower.keys())

    # Remove the topic names themselves from shared (avoid self-loops)
    shared_keys -= {topic_a.lower(), topic_b.lower()}

    if shared_keys:
        for key in list(shared_keys)[:10]:  # Limit to avoid visual clutter
            shared_label = entities_a_lower.get(key, key)
            sid = _add_node(shared_label, "shared", f"Shared concept: {shared_label}")

            edges.append({
                "from_id": id_a,
                "to_id": sid,
                "from_label": topic_a,
                "to_label": shared_label,
                "relationship": "relates to",
            })
            edges.append({
                "from_id": sid,
                "to_id": id_b,
                "from_label": shared_label,
                "to_label": topic_b,
                "relationship": "relates to",
            })
    else:
        # No real overlap found.
        # Return empty ([], []) so the caller can show a clean empty state.
        # Do NOT fabricate a '?' bridge node — that was misleading in a demo context.
        return [], []

    return nodes, edges



def render_pyvis_graph(
    nodes: list[dict],
    edges: list[dict],
    topic_a: str = "",
    topic_b: str = "",
) -> str:
    """Render an interactive pyvis graph as an HTML string.

    Args:
        nodes: List of node dicts with id, label, color, title.
        edges: List of edge dicts with from_id, to_id, relationship.
        topic_a: First topic name (for sizing).
        topic_b: Second topic name (for sizing).

    Returns:
        HTML string ready for st.components.v1.html()
    """
    if not nodes:
        return _empty_graph_html()

    from pyvis.network import Network

    # Create network
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        directed=True,
    )

    # Physics settings for a nice layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.008,
                "springLength": 180,
                "springConstant": 0.06,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 150
            }
        },
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {
                "size": 14,
                "face": "Inter, system-ui, sans-serif",
                "color": "#e6edf3"
            },
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.3)",
                "size": 8
            }
        },
        "edges": {
            "color": {
                "color": "#484f58",
                "highlight": "#7C5CFC",
                "hover": "#7C5CFC"
            },
            "font": {
                "size": 11,
                "color": "#8b949e",
                "face": "Inter, system-ui, sans-serif",
                "strokeWidth": 0
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.6
                }
            },
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.15
            },
            "width": 1.5
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    # Add nodes
    topic_labels = {topic_a.lower(), topic_b.lower()} if topic_a and topic_b else set()

    for node in nodes:
        is_topic = node["label"].lower() in topic_labels
        is_shared = node.get("source_type") == "shared"

        size = 35 if is_topic else (25 if is_shared else 18)
        shape = "dot"
        border_color = "#7C5CFC" if is_topic else (node["color"] if is_shared else "#30363d")

        net.add_node(
            node["id"],
            label=node["label"],
            color={
                "background": node["color"],
                "border": border_color,
                "highlight": {
                    "background": node["color"],
                    "border": "#7C5CFC",
                },
            },
            size=size,
            shape=shape,
            title=node.get("title", node["label"]),
        )

    # Add edges
    for edge in edges:
        edge_color = "#7C5CFC" if edge.get("relationship") == "gap" else "#484f58"
        dash = edge.get("relationship") == "gap"

        net.add_edge(
            edge["from_id"],
            edge["to_id"],
            label=edge.get("relationship", ""),
            color=edge_color,
            dashes=dash,
        )

    # Generate HTML
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
        dir=tempfile.gettempdir(),
    )
    tmp.close()
    net.save_graph(tmp.name)

    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()

    # Clean up temp file
    Path(tmp.name).unlink(missing_ok=True)

    return html


def _empty_graph_html() -> str:
    """Return HTML for an empty/no-results graph state."""
    return """
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        height: 300px;
        background: #0d1117;
        border-radius: 12px;
        border: 1px solid #21262d;
        color: #8b949e;
        font-family: Inter, system-ui, sans-serif;
        font-size: 16px;
    ">
        <div style="text-align: center;">
            <div style="font-size: 48px; margin-bottom: 12px;">🔍</div>
            <div>No connections found yet.</div>
            <div style="font-size: 13px; margin-top: 4px; color: #484f58;">
                Ingest more material to discover topic relationships.
            </div>
        </div>
    </div>
    """
