"""StudyMind — figure crops queued at ingest and captioned by the vision model (offline)."""

import asyncio
from pathlib import Path

import pytest
from sqlmodel import select

from core import concepts, config, figures, topics
from core.config import settings
from core.db import Chunk, ChunkMedia, PendingFigure, session
from core.ingest import ingest_pdf
from core.llm import llm
from tests.pdfs import figure_pdf

CAPTION = "A diagram where citrate converts to isocitrate in the Krebs cycle."


@pytest.fixture
def vision(monkeypatch):
    """Fake vision model: replies in order (an Exception is raised); records each call's kwargs."""
    state = {"replies": [], "calls": []}

    async def complete(prompt, system=None, **kw):
        state["calls"].append({"prompt": prompt, **kw})
        reply = state["replies"].pop(0) if state["replies"] else CAPTION
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete", complete)
    return state


def _pending():
    with session() as s:
        return s.exec(select(PendingFigure).order_by(PendingFigure.id)).all()


@pytest.mark.asyncio
async def test_ingest_pdf_queues_figures(tmp_path):
    result = await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert result["status"] == "success" and result["figures_queued"] == 2
    rows = _pending()
    assert len(rows) == 2 and {r.section_heading for r in rows} == {"Krebs Cycle"}
    assert all(Path(r.image_path).is_file() and Path(r.image_path).parent.parent == config.FIGURE_DIR for r in rows)
    assert figures.pending_count([result["source_id"]]) == 2


@pytest.mark.asyncio
async def test_caption_success_indexes_figure_chunk(tmp_path, vision):
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert await figures.caption_pending() == 2
    assert _pending() == []
    with session() as s:
        rows = s.exec(select(Chunk, ChunkMedia).join(ChunkMedia, ChunkMedia.chunk_id == Chunk.id)
                      .where(ChunkMedia.kind == "figure")).all()
    assert len(rows) == 2
    assert all(c.text == f"Figure (p.1): {CAPTION}" and c.page == 1 and m.kind == "figure"
               and Path(m.image_path).is_file() for c, m in rows)
    call = vision["calls"][0]
    assert call["model"] == settings.vision_model and call["images"][0].startswith(b"\x89PNG")
    assert "Krebs Cycle" in call["prompt"]


@pytest.mark.asyncio
async def test_short_caption_counts_attempts_and_drops_at_three(tmp_path, vision):
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    vision["replies"] = ["ok"] * 6
    seen = []
    for _ in range(3):
        assert await figures.caption_pending() == 0
        seen.append([r.attempts for r in _pending()])
    assert seen == [[1, 1], [2, 2], []]
    with session() as s:
        assert s.exec(select(ChunkMedia).where(ChunkMedia.kind == "figure")).first() is None


@pytest.mark.asyncio
async def test_llm_error_leaves_row_untouched(tmp_path, vision):
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    vision["replies"] = [ConnectionError("ollama down")]
    assert await figures.caption_pending() == 0
    assert [r.attempts for r in _pending()] == [0, 0] and len(vision["calls"]) == 1


@pytest.mark.asyncio
async def test_pending_for_deleted_source_is_dropped(tmp_path, vision):
    result = await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    topics.delete_source(result["source_id"])
    assert await figures.caption_pending() == 0
    assert _pending() == [] and vision["calls"] == []
    with session() as s:
        assert s.exec(select(Chunk)).first() is None


@pytest.mark.asyncio
async def test_worker_captions_before_concepts(monkeypatch):
    order = []

    async def fake_figures():
        order.append("figures")
        return 0

    async def fake_concepts():
        order.append("concepts")
        return 0

    monkeypatch.setattr(figures, "caption_pending", fake_figures)
    monkeypatch.setattr(concepts, "extract_pending", fake_concepts)
    concepts.start()
    await asyncio.sleep(0.05)
    await concepts.stop()
    assert order[:2] == ["figures", "concepts"]
