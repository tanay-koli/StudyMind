"""
StudyMind — Graph Utilities

Renders topic-connection graphs as interactive pyvis HTML.
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
