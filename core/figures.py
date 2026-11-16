"""
StudyMind — figure captioning.

Ingest queues each figure crop as a PendingFigure; the background worker sends it
to the vision model and indexes the caption as a chunk ("Figure (p.N): ...") with
a ChunkMedia row that links back to the image.
"""

import asyncio
import logging
from typing import Optional

from sqlmodel import delete, select

from core import config, retrieval
from core.config import check_llm, settings
from core.db import PendingFigure, Source, session
from core.llm import llm
from core.parse import Block, FigureCrop

log = logging.getLogger("studymind.figures")

MIN_CAPTION = 20
MAX_ATTEMPTS = 3
CAPTION_SYSTEM = "You describe figures from study material for a student."
CAPTION_PROMPT = "Describe this diagram for a student; list its labels and the relationships between them."


def _heading_for(page: int, blocks: list[Block]) -> Optional[str]:
    """Heading of the first block on the page, else of the last block before it."""
    on_page = [b for b in blocks if b.page == page]
    if on_page:
        return on_page[0].section_heading
    before = [b for b in blocks if b.page is not None and b.page < page]
    return before[-1].section_heading if before else None


def queue(source_id: int, topic_id: str, crops: list[FigureCrop], blocks: list[Block]) -> int:
    """Save the crops as PNGs and PendingFigure rows; returns how many were queued."""
    if not crops:
        return 0
    with session() as s:
        if s.get(Source, source_id) is None:  # deleted while the PDF was still being ingested
            return 0
    folder = config.FIGURE_DIR / str(source_id)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, crop in enumerate(crops):
        rel = f"{source_id}/p{crop.page}_{n}.png"  # relative to FIGURE_DIR, so moving the data folder keeps it
        (config.FIGURE_DIR / rel).write_bytes(crop.png)
        rows.append(PendingFigure(source_id=source_id, topic_id=topic_id, page=crop.page,
                                  section_heading=_heading_for(crop.page, blocks),
                                  bbox=",".join(f"{x:.1f}" for x in crop.bbox), image_path=rel))
    with session() as s:
        s.add_all(rows)
        s.commit()
    return len(rows)


async def caption_pending() -> int:
    """Caption every pending figure; returns how many became chunks. Skips the pass if the model is unreachable."""
    with session() as s:
        ids = s.exec(select(PendingFigure.id).order_by(PendingFigure.id)).all()
    if not ids:
        return 0
    status = await check_llm(settings.vision_model)
    if not status["ok"]:
        # Ollama down or the vision model not pulled: not the figures' fault, so no attempt is counted
        log.warning("vision model unavailable (%s); captioning resumes on the next pass", status["error"])
        return 0
    done = 0
    for fid in ids:
        with session() as s:
            row = s.get(PendingFigure, fid)
        if row is None:  # its source was deleted meanwhile
            continue
        try:
            png = (config.FIGURE_DIR / row.image_path).read_bytes()
        except OSError:
            log.warning("figure crop %s is missing; dropping it", row.image_path)
            _drop(fid)
            continue
        prompt = CAPTION_PROMPT + (
            f"\nIt appears under the heading: {row.section_heading}" if row.section_heading else ""
        )
        try:
            caption = " ".join((await llm.complete(prompt, system=CAPTION_SYSTEM, images=[png],
                                                   model=settings.vision_model, timeout=180)).split())
        except Exception as e:
            # The model is up (checked above), so this image is the problem: count it, don't block the queue
            log.warning("captioning %s failed: %s", row.image_path, e)
            caption = ""
        with session() as s:
            row = s.get(PendingFigure, fid)
            if row is None:
                continue
            if len(caption) < MIN_CAPTION:
                row.attempts += 1
                if row.attempts >= MAX_ATTEMPTS:
                    log.warning("no usable caption for %s after %d tries; dropping it", row.image_path, MAX_ATTEMPTS)
                    s.delete(row)
                else:
                    s.add(row)
                s.commit()
                continue
        block = Block(row.page, row.section_heading, f"Figure (p.{row.page}): {caption}")
        await asyncio.to_thread(retrieval.index_chunks, row.source_id, row.topic_id, [block],
                                figure={"image_path": row.image_path, "bbox": row.bbox})
        _drop(fid)
        done += 1
        # A study guide or audio script cached before this caption existed would never show the figure
        from core.topics import invalidate_caches  # topics → query → concepts → figures: import here
        invalidate_caches(row.topic_id)
    return done


def _drop(fid: int) -> None:
    with session() as s:
        s.exec(delete(PendingFigure).where(PendingFigure.id == fid))
        s.commit()
