"""
StudyMind — Topics Module (NotebookLM-style Notebooks)

A Topic is a named collection of ingested sources — like a Notebook.
Topics and sources are stored in SQLite (core/db.py). A topic's datasets
are the distinct Cognee dataset names of its sources.

Key design: when a source is ingested into a topic, the Cognee Cloud
datasetName IS the topic_id. This lets /api/v1/recall filter by
datasets=[topic_id] for scoped search.

Public API
----------
init_db()                                    (create tables, General, import topics.json)
load_topics()                                → dict
create_topic(name) → dict
delete_topic(topic_id)
get_topic_datasets(topic_id) → list[str]
add_source / list_sources / move_source / delete_source / delete_all_sources
suggest_topic_name(text_excerpt, existing_names) → str     (async)
generate_study_guide(topic_id, force=False) → str          (async)
generate_audio_script(topic_id, force=False) → str         (async)
synthesize_audio(topic_id, script) → Path | None
invalidate_caches(topic_id)                 → (clears study_guide + audio)
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import delete, select, update

from core import retrieval
from core.config import settings, DATA_DIR, _headers, http_client
from core.db import Chunk, Source, Topic, session
from core.llm import llm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEGACY_TOPICS_FILE = DATA_DIR / "topics.json"
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
}

_CACHE_FIELDS = ("study_guide", "study_guide_generated_at", "audio_script", "audio_generated_at")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables, import legacy data/topics.json once, ensure General exists."""
    with session() as s:
        if s.exec(select(Topic)).first() is None and LEGACY_TOPICS_FILE.exists():
            _import_legacy_json(s)
        if s.get(Topic, GENERAL_TOPIC_ID) is None:
            s.add(Topic(**GENERAL_TOPIC))
        s.commit()


def _import_legacy_json(s) -> None:
    """Copy v1 topics.json into the DB. The JSON file is left as a backup."""
    try:
        data = json.loads(LEGACY_TOPICS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    prefixes = {"pdf_": "pdf", "yt_": "youtube", "web_": "web", "note_": "note"}
    for tid, t in data.items():
        s.add(Topic(**{k: t.get(k) for k in ("id", "name", "created_at", "color", *_CACHE_FIELDS)}))
        for ds in t.get("datasets", []):
            stype = next((v for p, v in prefixes.items() if ds.startswith(p)), "unknown")
            s.add(Source(topic_id=tid, dataset_name=ds, source_type=stype, title=ds))


def load_topics() -> dict:
    """Return {topic_id: topic dict with a 'datasets' list}. Always includes 'general'."""
    with session() as s:
        topics = s.exec(select(Topic).order_by(Topic.created_at)).all()
        sources = s.exec(select(Source)).all()

    datasets: dict = {}
    for src in sources:
        ds = datasets.setdefault(src.topic_id, [])
        if src.dataset_name not in ds:
            ds.append(src.dataset_name)

    return {t.id: {**t.model_dump(), "datasets": datasets.get(t.id, [])} for t in topics}


def _update_topic(topic_id: str, **fields) -> None:
    with session() as s:
        topic = s.get(Topic, topic_id)
        if topic is None:
            return
        for k, v in fields.items():
            setattr(topic, k, v)
        s.add(topic)
        s.commit()


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
    """Create a new topic.

    Returns the new topic dict. If a topic with the same name (or slug)
    already exists, returns the existing one without modification.
    """
    topics = load_topics()
    tid = _topic_id(name)

    # Check for duplicate name (case-insensitive)
    for existing in topics.values():
        if existing["name"].lower() == name.strip().lower():
            return existing
    if tid in topics:
        return topics[tid]

    topic = Topic(id=tid, name=name.strip(), color=_next_color(topics))
    with session() as s:
        s.add(topic)
        s.commit()
    return {**topic.model_dump(), "datasets": []}


def delete_topic(topic_id: str) -> bool:
    """Delete a topic and its source rows. Returns False if not found or General."""
    if topic_id == GENERAL_TOPIC_ID:
        return False  # Cannot delete General
    with session() as s:
        topic = s.get(Topic, topic_id)
        if topic is None:
            return False
        _purge(s, s.exec(select(Source.id).where(Source.topic_id == topic_id)).all())
        s.exec(delete(Source).where(Source.topic_id == topic_id))
        s.delete(topic)
        s.commit()
    return True


def get_topic_datasets(topic_id: str) -> list:
    """Return list of dataset names for a topic."""
    with session() as s:
        names = s.exec(select(Source.dataset_name).where(Source.topic_id == topic_id)).all()
    return list(dict.fromkeys(names))


def invalidate_caches(topic_id: str) -> None:
    """Clear cached study guide and audio for a topic."""
    _update_topic(topic_id, **dict.fromkeys(_CACHE_FIELDS))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _purge(s, source_ids: list[int]) -> None:
    """Delete the sources' Qdrant points and chunk rows (caller commits)."""
    retrieval.delete_source_points(list(source_ids))
    s.exec(delete(Chunk).where(Chunk.source_id.in_(source_ids)))


def add_source(topic_id: str, dataset_name: str, source_type: str, title: str) -> dict:
    """Record an ingested source and invalidate the topic's caches."""
    src = Source(topic_id=topic_id, dataset_name=dataset_name, source_type=source_type, title=title)
    with session() as s:
        s.add(src)
        s.commit()
    invalidate_caches(topic_id)
    return src.model_dump()


def list_sources(topic_id: Optional[str] = None) -> list[dict]:
    with session() as s:
        q = select(Source).order_by(Source.created_at)
        if topic_id:
            q = q.where(Source.topic_id == topic_id)
        return [src.model_dump() for src in s.exec(q).all()]


def move_source(source_id: int, topic_id: str) -> Optional[dict]:
    """Reassign a source and its chunks to another topic (the Cognee dataset stays the same)."""
    with session() as s:
        src = s.get(Source, source_id)
        if src is None or s.get(Topic, topic_id) is None:
            return None
        old_topic = src.topic_id
        src.topic_id = topic_id
        s.add(src)
        s.exec(update(Chunk).where(Chunk.source_id == source_id).values(topic_id=topic_id))
        retrieval.move_source_points(source_id, topic_id)
        s.commit()
    invalidate_caches(old_topic)
    invalidate_caches(topic_id)
    return src.model_dump()


def delete_source(source_id: int) -> Optional[dict]:
    """Remove a source, its chunks and Qdrant points. Cognee Cloud data is not deleted (no delete API yet)."""
    with session() as s:
        src = s.get(Source, source_id)
        if src is None:
            return None
        _purge(s, [source_id])
        s.delete(src)
        s.commit()
    invalidate_caches(src.topic_id)
    return src.model_dump()


def delete_all_sources() -> int:
    with session() as s:
        topic_ids = s.exec(select(Topic.id)).all()
        source_ids = s.exec(select(Source.id)).all()
        count = len(source_ids)
        _purge(s, source_ids)
        s.exec(delete(Source))
        s.commit()
    for tid in topic_ids:
        invalidate_caches(tid)
    return count


# ---------------------------------------------------------------------------
# Internal LLM helper — single /api/v1/recall GRAPH_COMPLETION call
# ---------------------------------------------------------------------------

async def _llm_call(prompt: str, topic_id: Optional[str] = None, timeout: int = 60) -> str:
    """Make one GRAPH_COMPLETION recall call and return the text response.

    If topic_id is provided, scopes the search to that dataset.
    Returns empty string on failure.
    """
    if not settings.cognee_base_url or not settings.cognee_api_key:
        return ""

    payload: dict = {
        "query": prompt,
        "search_type": "GRAPH_COMPLETION",
        "top_k": 8,
    }
    if topic_id and topic_id != GENERAL_TOPIC_ID:
        payload["datasets"] = [topic_id]

    try:
        resp = await http_client().post(
            f"{settings.cognee_base_url}/api/v1/recall",
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

async def suggest_topic_name(text_excerpt: str, existing_names: list[str]) -> str:
    """Suggest a 2-4 word topic name for the given text excerpt.

    Needs no retrieval, so it calls the LLM directly. Returns empty string
    if the call fails or the response is uninformative.
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

    try:
        # Ollama's first call after idle loads the model (~1 min on CPU)
        result = await llm.complete(prompt, timeout=120)
    except Exception:
        return ""
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

async def generate_study_guide(topic_id: str, force: bool = False) -> str:
    """Generate a markdown study guide for the topic.

    Uses the cached version unless force=True or cache is empty.
    Makes ONE GRAPH_COMPLETION recall call.

    Returns the markdown string, or an error message.
    """
    topic = load_topics().get(topic_id)
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

    guide = await _llm_call(prompt, topic_id=topic_id, timeout=90)

    if not guide:
        guide = (
            f"# Study Guide: {topic_name}\n\n"
            "_Could not generate study guide — no content found in the knowledge graph yet. "
            "Make sure your sources have been successfully ingested._"
        )

    _update_topic(topic_id, study_guide=guide, study_guide_generated_at=datetime.now().isoformat())
    return guide


# ---------------------------------------------------------------------------
# Feature: Audio Overview script generation
# ---------------------------------------------------------------------------

async def generate_audio_script(topic_id: str, force: bool = False) -> str:
    """Generate a podcast-style audio script for the topic.

    Cached on the topic row. Makes ONE GRAPH_COMPLETION recall call.
    Returns the script string.
    """
    topic = load_topics().get(topic_id)
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

    script = await _llm_call(prompt, topic_id=topic_id, timeout=90)

    if not script:
        script = (
            f"Welcome to your audio overview of {topic_name}. "
            "No content was found in the knowledge graph yet. "
            "Please ingest some study material and try again."
        )

    _update_topic(topic_id, audio_script=script, audio_generated_at=datetime.now().isoformat())
    return script


# ---------------------------------------------------------------------------
# Feature: Audio synthesis via pyttsx3
# ---------------------------------------------------------------------------

def synthesize_audio(topic_id: str, script: str) -> Optional[Path]:
    """Convert a script string to a WAV audio file using pyttsx3.

    Blocking — call it via asyncio.to_thread from async code.
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
