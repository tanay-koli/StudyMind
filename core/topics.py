"""
StudyMind — Topics Module (NotebookLM-style Notebooks)

A Topic is a named collection of ingested sources — like a Notebook.
Topics are stored locally in data/topics.json and each maps to one or
more Cognee Cloud dataset names.

Key design: when a source is ingested into a topic, the Cognee Cloud
datasetName IS the topic_id. This lets /api/v1/recall filter by
datasets=[topic_id] for scoped search.

Public API
----------
load_topics()                                → dict
save_topics(topics)
create_topic(name) → dict
delete_topic(topic_id)
add_dataset_to_topic(topic_id, dataset_name)
remove_dataset_from_topic(topic_id, dataset_name)
get_topic_datasets(topic_id) → list[str]
suggest_topic_name(text_excerpt, existing_names) → str
generate_study_guide(topic_id, force=False) → str
generate_audio_script(topic_id, force=False) → str
synthesize_audio(topic_id, script) → Path | None
invalidate_caches(topic_id)                 → (clears study_guide + audio)
"""

import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from core.config import COGNEE_BASE_URL, COGNEE_API_KEY, DATA_DIR, _headers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPICS_FILE = DATA_DIR / "topics.json"
AUDIO_DIR   = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

COLOR_PALETTE = [
    "#7C5CFC",  # Purple
    "#00C9A7",  # Teal
    "#FF6B6B",  # Coral
    "#FFA502",  # Amber
    "#1E90FF",  # Blue
    "#2ED573",  # Green
]

GENERAL_TOPIC_ID = "general"
GENERAL_TOPIC = {
    "id": GENERAL_TOPIC_ID,
    "name": "General",
    "created_at": "2026-01-01T00:00:00",
    "color": "#A4B0BE",
    "datasets": [],
    "study_guide": None,
    "study_guide_generated_at": None,
    "audio_script": None,
    "audio_generated_at": None,
}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def load_topics() -> dict:
    """Load topics from data/topics.json. Always includes 'general'."""
    if TOPICS_FILE.exists():
        try:
            data = json.loads(TOPICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    # Ensure General always exists
    if GENERAL_TOPIC_ID not in data:
        data[GENERAL_TOPIC_ID] = dict(GENERAL_TOPIC)

    return data


def save_topics(topics: dict) -> None:
    """Persist topics dict to data/topics.json."""
    TOPICS_FILE.write_text(
        json.dumps(topics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _next_color(topics: dict) -> str:
    """Pick the next color from the palette (cycles)."""
    used = [t.get("color") for t in topics.values()]
    for color in COLOR_PALETTE:
        if color not in used:
            return color
    # All used — cycle by count
    return COLOR_PALETTE[len(topics) % len(COLOR_PALETTE)]


def _topic_id(name: str) -> str:
    """Derive a stable slug ID from a topic name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    return slug or hashlib.md5(name.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Topic CRUD
# ---------------------------------------------------------------------------

def create_topic(name: str) -> dict:
    """Create a new topic and save to topics.json.

    Returns the new topic dict. If a topic with the same name already
    exists, returns the existing one without modification.
    """
    topics = load_topics()
    tid = _topic_id(name)

    # Check for duplicate name (case-insensitive)
    for existing in topics.values():
        if existing["name"].lower() == name.strip().lower():
            return existing

    topic = {
        "id": tid,
        "name": name.strip(),
        "created_at": datetime.now().isoformat(),
        "color": _next_color(topics),
        "datasets": [],
        "study_guide": None,
        "study_guide_generated_at": None,
        "audio_script": None,
        "audio_generated_at": None,
    }
    topics[tid] = topic
    save_topics(topics)
    return topic


def delete_topic(topic_id: str) -> bool:
    """Delete a topic by ID. Returns True if deleted, False if not found."""
    if topic_id == GENERAL_TOPIC_ID:
        return False  # Cannot delete General
    topics = load_topics()
    if topic_id not in topics:
        return False
    del topics[topic_id]
    save_topics(topics)
    return True


def add_dataset_to_topic(topic_id: str, dataset_name: str) -> None:
    """Register a dataset_name under a topic and invalidate caches."""
    topics = load_topics()
    if topic_id not in topics:
        return
    if dataset_name not in topics[topic_id]["datasets"]:
        topics[topic_id]["datasets"].append(dataset_name)
    # Invalidate cached study guide & audio since content changed
    topics[topic_id]["study_guide"] = None
    topics[topic_id]["study_guide_generated_at"] = None
    topics[topic_id]["audio_script"] = None
    topics[topic_id]["audio_generated_at"] = None
    save_topics(topics)


def remove_dataset_from_topic(topic_id: str, dataset_name: str) -> None:
    """Remove a dataset_name from a topic."""
    topics = load_topics()
    if topic_id not in topics:
        return
    if dataset_name in topics[topic_id]["datasets"]:
        topics[topic_id]["datasets"].remove(dataset_name)
        topics[topic_id]["study_guide"] = None
        topics[topic_id]["audio_script"] = None
        save_topics(topics)


def get_topic_datasets(topic_id: str) -> list:
    """Return list of dataset names for a topic."""
    topics = load_topics()
    topic = topics.get(topic_id, {})
    return topic.get("datasets", [])


def invalidate_caches(topic_id: str) -> None:
    """Clear cached study guide and audio for a topic."""
    topics = load_topics()
    if topic_id in topics:
        topics[topic_id]["study_guide"] = None
        topics[topic_id]["study_guide_generated_at"] = None
        topics[topic_id]["audio_script"] = None
        topics[topic_id]["audio_generated_at"] = None
        save_topics(topics)


# ---------------------------------------------------------------------------
# Internal LLM helper — single /api/v1/recall GRAPH_COMPLETION call
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, topic_id: Optional[str] = None, timeout: int = 60) -> str:
    """Make one GRAPH_COMPLETION recall call and return the text response.

    If topic_id is provided, scopes the search to that dataset.
    Returns empty string on failure.
    """
    if not COGNEE_BASE_URL or not COGNEE_API_KEY:
        return ""

    payload: dict = {
        "query": prompt,
        "search_type": "GRAPH_COMPLETION",
        "top_k": 8,
    }
    if topic_id and topic_id != GENERAL_TOPIC_ID:
        payload["datasets"] = [topic_id]

    try:
        resp = httpx.post(
            f"{COGNEE_BASE_URL}/api/v1/recall",
            headers=_headers(),
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            return _extract_text(data)
        return ""
    except Exception:
        return ""


def _extract_text(data) -> str:
    """Pull text from a Cognee recall response."""
    if not data:
        return ""
    if isinstance(data, str):
        return data

    texts = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            t = (
                item.get("answer", "")
                or item.get("text", "")
                or item.get("content", "")
                or item.get("name", "")
            )
            if t:
                texts.append(str(t).strip())
    return "\n\n".join(t for t in texts if t)


# ---------------------------------------------------------------------------
# Feature: AI topic suggestion
# ---------------------------------------------------------------------------

def suggest_topic_name(text_excerpt: str, existing_names: list[str]) -> str:
    """Suggest a 2-4 word topic name for the given text excerpt.

    Makes ONE GRAPH_COMPLETION recall call. Returns empty string if the
    call fails or the response is uninformative.
    """
    excerpt = text_excerpt[:500].strip()
    existing_str = ", ".join(existing_names) if existing_names else "none"

    prompt = (
        f"Given this text excerpt, suggest a short topic name (2-4 words) "
        f"that best describes what it's about.\n"
        f"Existing topics: {existing_str}.\n"
        f"If the content fits an existing topic, suggest that exact name.\n"
        f"Reply with ONLY the topic name, nothing else.\n\n"
        f"Text:\n{excerpt}"
    )

    result = _llm_call(prompt, topic_id=None, timeout=30)
    # Clean: strip quotes, punctuation, extra whitespace, take first line
    suggestion = result.strip().split("\n")[0].strip(' "\'.,')
    # Validate: 2-6 words, not too long
    words = suggestion.split()
    if 1 <= len(words) <= 6 and len(suggestion) <= 60:
        return suggestion
    return ""


# ---------------------------------------------------------------------------
# Feature: Study Guide generation
# ---------------------------------------------------------------------------

def generate_study_guide(topic_id: str, force: bool = False) -> str:
    """Generate a markdown study guide for the topic.

    Uses cached version from topics.json unless force=True or cache is empty.
    Makes ONE GRAPH_COMPLETION recall call.

    Returns the markdown string, or an error message.
    """
    topics = load_topics()
    topic = topics.get(topic_id)
    if not topic:
        return f"Topic '{topic_id}' not found."

    # Return cached version if available and not forced
    if not force and topic.get("study_guide"):
        return topic["study_guide"]

    if not topic.get("datasets"):
        return (
            f"No sources ingested into **{topic['name']}** yet. "
            "Add some study material first using the Ingest tab."
        )

    topic_name = topic["name"]
    prompt = (
        f"Create a study guide for the topic '{topic_name}' from the material in this knowledge base.\n\n"
        f"Structure the guide exactly as follows:\n"
        f"## Key Concepts\n"
        f"(bullet list of 5-8 main ideas)\n\n"
        f"## Important Details\n"
        f"(2-3 sentences per concept from the Key Concepts section)\n\n"
        f"## Quick Review Questions\n"
        f"(3-5 questions to test understanding)\n\n"
        f"Be specific and accurate. Base everything on the actual study material."
    )

    guide = _llm_call(prompt, topic_id=topic_id, timeout=90)

    if not guide:
        guide = (
            f"# Study Guide: {topic_name}\n\n"
            "_Could not generate study guide — no content found in the knowledge graph yet. "
            "Make sure your sources have been successfully ingested._"
        )

    # Cache in topics.json
    topics[topic_id]["study_guide"] = guide
    topics[topic_id]["study_guide_generated_at"] = datetime.now().isoformat()
    save_topics(topics)

    return guide


# ---------------------------------------------------------------------------
# Feature: Audio Overview script generation
# ---------------------------------------------------------------------------

def generate_audio_script(topic_id: str, force: bool = False) -> str:
    """Generate a podcast-style audio script for the topic.

    Cached in topics.json. Makes ONE GRAPH_COMPLETION recall call.
    Returns the script string.
    """
    topics = load_topics()
    topic = topics.get(topic_id)
    if not topic:
        return f"Topic '{topic_id}' not found."

    if not force and topic.get("audio_script"):
        return topic["audio_script"]

    if not topic.get("datasets"):
        return (
            f"No sources in '{topic['name']}' yet. "
            "Add study material first."
        )

    topic_name = topic["name"]
    prompt = (
        f"Write a 2-minute podcast-style overview of '{topic_name}' "
        f"from the material in this knowledge base.\n\n"
        f"Use a warm, conversational tone — like explaining to a curious friend.\n"
        f"Structure:\n"
        f"1. Hook: one compelling sentence that grabs attention\n"
        f"2. Three main points from the material\n"
        f"3. Closing takeaway: one memorable sentence\n\n"
        f"Keep it under 300 words. No bullet points — write in natural spoken sentences."
    )

    script = _llm_call(prompt, topic_id=topic_id, timeout=90)

    if not script:
        script = (
            f"Welcome to your audio overview of {topic_name}. "
            "No content was found in the knowledge graph yet. "
            "Please ingest some study material and try again."
        )

    # Cache in topics.json
    topics[topic_id]["audio_script"] = script
    topics[topic_id]["audio_generated_at"] = datetime.now().isoformat()
    save_topics(topics)

    return script


# ---------------------------------------------------------------------------
# Feature: Audio synthesis via pyttsx3
# ---------------------------------------------------------------------------

def synthesize_audio(topic_id: str, script: str) -> Optional[Path]:
    """Convert a script string to a WAV audio file using pyttsx3.

    Saves to data/audio/{topic_id}_overview.wav.
    Returns the Path on success, None on failure.
    """
    out_path = AUDIO_DIR / f"{topic_id}_overview.wav"

    try:
        import pyttsx3  # noqa: PLC0415

        engine = pyttsx3.init()
        engine.setProperty("rate", 165)    # words per minute
        engine.setProperty("volume", 0.9)

        engine.save_to_file(script, str(out_path))
        engine.runAndWait()
        engine.stop()

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        return None

    except ImportError:
        return None
    except Exception:
        return None
