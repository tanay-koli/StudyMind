"""
StudyMind — Data Ingestion

Async functions to ingest study material from four source types. Each function:
  1. Parses the source locally (PDFs keep page and section heading)
  2. Records the source in SQLite, chunks it and indexes the chunks in Qdrant
  3. Returns a status dict for the UI
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from core import concepts, retrieval
from core.chunking import chunk
from core.parse import Block, parse_pdf, text_blocks
from core.topics import GENERAL_TOPIC_ID, add_source, delete_source


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _register_in_topic(result: dict) -> None:
    """Record a successful ingestion as a source row; adds source_id to result."""
    src = add_source(result["topic_id"], result["source_type"], result["title"])
    result["source_id"] = src["id"]


async def _store(result: dict, blocks: list[Block]) -> None:
    """Register the source and index its chunks; on failure leave nothing behind."""
    _register_in_topic(result)
    sid = result["source_id"]
    try:
        result["chunk_count"] = await asyncio.to_thread(
            lambda: retrieval.index_chunks(sid, result["topic_id"], chunk(blocks))
        )
    except Exception as exc:
        # Roll back so a failed source never half-exists (also purges its chunks)
        delete_source(sid)
        result.pop("source_id")
        result["status"] = "error"
        result["error"] = f"Indexing failed: {type(exc).__name__}: {exc}"
        return

    result["status"] = "success"
    concepts.wake()
    result["excerpt"] = (result["title"] + "\n\n" + "\n\n".join(b.text for b in blocks))[:500]


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------

async def ingest_pdf(
    file_path: str,
    title: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a PDF file into the local index.

    Args:
        file_path: Path to the PDF file on disk.
        title: Optional human-readable title. Defaults to filename.
        topic_id: Optional topic to assign this source to.

    Returns:
        dict: source_type, title, topic_id, status, error, timestamp, source_id, chunk_count
    """
    if title is None:
        title = Path(file_path).stem

    result = {
        "source_type": "pdf",
        "title": title,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        blocks = await asyncio.to_thread(parse_pdf, file_path)
        if not blocks:
            result["status"] = "error"
            result["error"] = "PDF contains no extractable text."
            return result

        result["page_count"] = len({b.page for b in blocks})
        await _store(result, blocks)

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_youtube(
    url: str,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a YouTube video transcript into the local index.

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
            "topic_id": topic_id or GENERAL_TOPIC_ID,
            "status": "error",
            "error": f"Could not extract video ID from URL: {url}",
            "timestamp": datetime.now().isoformat(),
        }


    result = {
        "source_type": "youtube",
        "title": f"YouTube: {video_id}",
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "url": url,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = await asyncio.to_thread(ytt_api.fetch, video_id)

        lines = []
        for snippet in transcript:
            text = snippet.text if hasattr(snippet, "text") else str(snippet.get("text", ""))
            lines.append(text)

        if not lines:
            result["status"] = "error"
            result["error"] = "No transcript available for this video."
            return result

        await _store(result, text_blocks(" ".join(lines)))

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_url(
    url: str,
    title: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest a web article into the local index.

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


    result = {
        "source_type": "web",
        "title": title,
        "topic_id": topic_id or GENERAL_TOPIC_ID,
        "url": url,
        "status": "pending",
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)
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

        await _store(result, text_blocks(extracted))

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result


async def ingest_text(
    text: str,
    title: str = "Untitled Note",
    topic_id: Optional[str] = None,
) -> dict:
    """Ingest typed/pasted notes into the local index.

    Args:
        text: The note content.
        title: Title for the note.
        topic_id: Optional topic to assign this source to.

    Returns:
        Status dict.
    """

    result = {
        "source_type": "note",
        "title": title,
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
        await _store(result, text_blocks(text))

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"Ingestion failed: {type(exc).__name__}: {exc}"

    return result
