"""
StudyMind — Data Ingestion (Cognee Cloud)

Async-compatible functions to ingest study material from four source types
into the Cognee Cloud knowledge graph. Each function:
  1. Parses the source into plain text (locally, unchanged)
  2. POSTs the text as a file upload to POST /api/v1/remember
     using topic_id as the datasetName (enables scoped recall)
  3. Registers the source in topics.json
  4. Returns a status dict for the UI

The Cognee Cloud handles chunking, embedding, and graph construction
server-side — no local LLM or DB required.
"""

import re
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import httpx

from core.config import COGNEE_BASE_URL, COGNEE_API_KEY, DATA_DIR, _headers
from core.topics import (
    GENERAL_TOPIC_ID,
    load_topics,
    create_topic,
    add_dataset_to_topic,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_dataset_name(name: str) -> str:
    """Convert a human-readable name into a valid Cognee dataset name."""
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    sanitized = sanitized.lower()[:60]
    return sanitized or "untitled"


def _make_dataset_name(source_type: str, title: str) -> str:
    """Generate a unique dataset name from source type and title."""
    return f"{source_type}_{_sanitize_dataset_name(title)}"


def _upload_to_cloud(
    text: str,
    dataset_name: str,
    timeout: int = 60,
) -> dict:
    """Upload plain text to Cognee Cloud via POST /api/v1/remember.

    Args:
        text: The full text content to ingest.
        dataset_name: Cognee datasetName (topic_id when topic-aware).
        timeout: HTTP timeout in seconds.

    Returns:
        dict: ok (bool), error (str|None), status_code (int|None)
    """
    if not COGNEE_BASE_URL or not COGNEE_API_KEY:
        return {
            "ok": False,
            "error": "Cognee Cloud credentials not configured. Check .env.",
            "status_code": None,
        }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            resp = httpx.post(
                f"{COGNEE_BASE_URL}/api/v1/remember",
                headers=_headers(),
                files={"data": (f"{dataset_name}.txt", f, "text/plain")},
                data={"datasetName": dataset_name},
                timeout=timeout,
            )

        if resp.status_code in (200, 201, 202):
            return {"ok": True, "error": None, "status_code": resp.status_code}

        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:300]
        return {
            "ok": False,
            "error": f"Cognee Cloud returned HTTP {resp.status_code}: {detail}",
            "status_code": resp.status_code,
        }

    except httpx.ConnectError:
        return {
            "ok": False,
            "error": (
                f"Cannot reach Cognee Cloud at {COGNEE_BASE_URL}. "
                "Check your internet connection."
            ),
            "status_code": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Upload failed: {type(exc).__name__}: {exc}",
            "status_code": None,
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _resolve_upload_dataset(topic_id: Optional[str], fallback_dataset: str) -> str:
    """Determine which Cognee datasetName to use for the upload.

    When topic_id is provided (and not 'general'), use the topic_id directly
    as the datasetName so /api/v1/recall can filter by datasets=[topic_id].
    For 'general' or no topic, use the source-type-based dataset name.
    """
    if topic_id and topic_id != GENERAL_TOPIC_ID:
        return topic_id
    return fallback_dataset


def _register_in_topic(topic_id: Optional[str], dataset_name: str) -> None:
    """Add dataset_name to the topic in topics.json. Creates topic if needed."""
    tid = topic_id or GENERAL_TOPIC_ID
    topics = load_topics()
    if tid not in topics:
        # Shouldn't happen — General always exists — but guard anyway
        create_topic("General" if tid == GENERAL_TOPIC_ID else tid)
    add_dataset_to_topic(tid, dataset_name)


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------

async def ingest_pdf(
    file_path: str,
    title: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a PDF file into the Cognee Cloud knowledge graph.

    Args:
        file_path: Path to the PDF file on disk.
        title: Optional human-readable title. Defaults to filename.
        topic_id: Optional topic to assign this source to. Uses topic_id
                  as the Cognee datasetName for scoped recall.

    Returns:
        dict: source_type, title, dataset_name, topic_id, status, error, timestamp
    """
    import fitz  # PyMuPDF

    if title is None:
        title = Path(file_path).stem

    fallback_dataset = _make_dataset_name("pdf", title)
    upload_dataset   = _resolve_upload_dataset(topic_id, fallback_dataset)

    result = {
        "source_type": "pdf",
        "title": title,
        "dataset_name": upload_dataset,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "file_path": file_path,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        doc = fitz.open(file_path)
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()

        if not pages_text:
            result["status"] = "error"
            result["error"] = "PDF contains no extractable text."
            return result

        full_text = f"Source: PDF — {title}\n\n" + "\n\n".join(pages_text)

        upload = _upload_to_cloud(full_text, upload_dataset)
        if upload["ok"]:
            result["status"] = "success"
            result["page_count"] = len(pages_text)
            result["excerpt"] = full_text[:500]
            _register_in_topic(topic_id, upload_dataset)
        else:
            result["status"] = "error"
            result["error"] = upload["error"]

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_youtube(
    url: str,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a YouTube video transcript into the Cognee Cloud knowledge graph.

    Args:
        url: YouTube video URL.
        topic_id: Optional topic to assign this source to.

    Returns:
        Status dict.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = None
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    elif "v=" in url:
        video_id = url.split("v=")[-1].split("&")[0]
    elif "shorts/" in url:
        video_id = url.split("shorts/")[-1].split("?")[0]

    if not video_id:
        return {
            "source_type": "youtube",
            "title": url,
            "dataset_name": "",
            "topic_id": topic_id or GENERAL_TOPIC_ID,
            "status": "error",
            "error": f"Could not extract video ID from URL: {url}",
            "timestamp": datetime.now().isoformat(),
        }

    fallback_dataset = _make_dataset_name("yt", video_id)
    upload_dataset   = _resolve_upload_dataset(topic_id, fallback_dataset)

    result = {
        "source_type": "youtube",
        "title": f"YouTube: {video_id}",
        "dataset_name": upload_dataset,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "url": url,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        lines = []
        for snippet in transcript:
            text = snippet.text if hasattr(snippet, "text") else str(snippet.get("text", ""))
            lines.append(text)

        if not lines:
            result["status"] = "error"
            result["error"] = "No transcript available for this video."
            return result

        full_text = (
            f"Source: YouTube Video — {video_id}\nURL: {url}\n\n"
            + " ".join(lines)
        )

        upload = _upload_to_cloud(full_text, upload_dataset)
        if upload["ok"]:
            result["status"] = "success"
            result["excerpt"] = full_text[:500]
            _register_in_topic(topic_id, upload_dataset)
        else:
            result["status"] = "error"
            result["error"] = upload["error"]

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_url(
    url: str,
    title: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a web article into the Cognee Cloud knowledge graph.

    Args:
        url: Web article URL.
        title: Optional title override.
        topic_id: Optional topic to assign this source to.

    Returns:
        Status dict.
    """
    import trafilatura

    if title is None:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        title = f"{parsed.netloc}{parsed.path}"[:80]

    fallback_dataset = _make_dataset_name("web", title)
    upload_dataset   = _resolve_upload_dataset(topic_id, fallback_dataset)

    result = {
        "source_type": "web",
        "title": title,
        "dataset_name": upload_dataset,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "url": url,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            result["status"] = "error"
            result["error"] = f"Could not fetch content from {url}"
            return result

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )

        if not extracted or not extracted.strip():
            result["status"] = "error"
            result["error"] = "Could not extract meaningful text from the page."
            return result

        metadata = trafilatura.extract_metadata(downloaded)
        if metadata and metadata.title:
            result["title"] = metadata.title
            # Note: keep upload_dataset as topic_id for scoping
            result["dataset_name"] = upload_dataset

        full_text = (
            f"Source: Web Article — {result['title']}\nURL: {url}\n\n" + extracted
        )

        upload = _upload_to_cloud(full_text, upload_dataset)
        if upload["ok"]:
            result["status"] = "success"
            result["excerpt"] = full_text[:500]
            _register_in_topic(topic_id, upload_dataset)
        else:
            result["status"] = "error"
            result["error"] = upload["error"]

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_text(
    text: str,
    title: str = "Untitled Note",
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest typed/pasted notes into the Cognee Cloud knowledge graph.

    Args:
        text: The note content.
        title: Title for the note.
        topic_id: Optional topic to assign this source to.

    Returns:
        Status dict.
    """
    fallback_dataset = _make_dataset_name("note", title)
    upload_dataset   = _resolve_upload_dataset(topic_id, fallback_dataset)

    result = {
        "source_type": "note",
        "title": title,
        "dataset_name": upload_dataset,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    if not text or not text.strip():
        result["status"] = "error"
        result["error"] = "Note is empty."
        return result

    try:
        full_text = f"Source: Personal Note — {title}\n\n{text}"

        upload = _upload_to_cloud(full_text, upload_dataset)
        if upload["ok"]:
            result["status"] = "success"
            result["excerpt"] = full_text[:500]
            _register_in_topic(topic_id, upload_dataset)
        else:
            result["status"] = "error"
            result["error"] = upload["error"]

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def forget_source(dataset_name: str) -> dict:
    """Remove a dataset from session tracking.

    Note: No cloud delete API is available. Session-only removal.
    """
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "note": "Source removed from session. Data persists in Cognee Cloud.",
    }


async def forget_all() -> dict:
    """Remove ALL sources from session tracking."""
    return {
        "status": "success",
        "note": "Session cleared. Data persists in Cognee Cloud.",
    }
