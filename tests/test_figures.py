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

    async def reachable(model=None, timeout=5):
        return {"ok": state.get("reachable", True), "error": None if state.get("reachable", True) else "down", "model": model}

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(figures, "check_llm", reachable)
    return state


def _chunks():
    with session() as s:
        return s.exec(select(Chunk).order_by(Chunk.id)).all()


def _pending():
    with session() as s:
        return s.exec(select(PendingFigure).order_by(PendingFigure.id)).all()


@pytest.mark.asyncio
async def test_ingest_pdf_queues_figures(tmp_path):
    result = await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert result["status"] == "success" and result["figures_queued"] == 2
    rows = _pending()
    assert len(rows) == 2 and {r.section_heading for r in rows} == {"Krebs Cycle"}
    assert all((config.FIGURE_DIR / r.image_path).is_file() and r.image_path.startswith(f"{r.source_id}/") for r in rows)


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
               and (config.FIGURE_DIR / m.image_path).is_file() for c, m in rows)
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
async def test_unreachable_vision_model_skips_the_pass(tmp_path, vision):
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    vision["reachable"] = False
    assert await figures.caption_pending() == 0
    assert [r.attempts for r in _pending()] == [0, 0] and vision["calls"] == []


@pytest.mark.asyncio
async def test_one_failing_figure_does_not_block_the_rest(tmp_path, vision):
    # e.g. a 180 s timeout or an Ollama 500 on one image, with the model otherwise fine
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    vision["replies"] = [RuntimeError("500 from ollama"), CAPTION]
    assert await figures.caption_pending() == 1
    assert [r.attempts for r in _pending()] == [1]


@pytest.mark.asyncio
async def test_reused_ids_do_not_give_a_new_upload_the_old_caption(tmp_path, vision, monkeypatch):
    """Delete A while its figure is being captioned, upload B: SQLite would reuse A's ids for B."""
    from tests.pdfs import diagram_only_pdf

    a = await ingest_pdf(str(diagram_only_pdf(tmp_path / "a.pdf")))

    async def complete(prompt, system=None, **kw):
        if not any(c.text.startswith("Figure") for c in _chunks()) and _pending()[0].source_id == a["source_id"]                 and not vision.get("swapped"):
            vision["swapped"] = True
            topics.delete_source(a["source_id"])
            await ingest_pdf(str(diagram_only_pdf(tmp_path / "b.pdf")))
            return "Caption of figure A: citrate becomes isocitrate."
        return "Caption of figure B: citrate becomes isocitrate."

    monkeypatch.setattr(llm, "complete", complete)
    await figures.caption_pending()
    await figures.caption_pending()
    texts = [c.text for c in _chunks() if c.text.startswith("Figure")]
    assert texts == ["Figure (p.2): Caption of figure B: citrate becomes isocitrate."] and _pending() == []


@pytest.mark.asyncio
async def test_crops_survive_moving_the_data_folder(tmp_path, vision, monkeypatch):
    import shutil

    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert not any(Path(r.image_path).is_absolute() for r in _pending())
    moved = tmp_path / "moved_figures"
    shutil.move(str(config.FIGURE_DIR), moved)
    monkeypatch.setattr(config, "FIGURE_DIR", moved)
    assert await figures.caption_pending() == 2


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


@pytest.mark.asyncio
async def test_diagram_question_returns_figure_and_page(tmp_path, vision):
    """Done-when (offline): a question only the diagram answers cites the figure on page 2."""
    from core import agent, query
    from tests.pdfs import diagram_only_pdf

    caption = "Flow diagram: citrate is converted to isocitrate by aconitase."
    vision["replies"] = [caption]
    await ingest_pdf(str(diagram_only_pdf(tmp_path / "d.pdf")))
    assert await figures.caption_pending() == 1

    res = await query.ask("what converts citrate to isocitrate?")
    top = res["citations"][0]
    assert (top["page"], top["kind"]) == (2, "figure") and top["image_url"].startswith("/figures/")

    async with agent.open_graph(str(tmp_path / "a.db")) as g:
        cfg = {"configurable": {"thread_id": "t"}}
        await g.ainvoke({"messages": [], "question": "what converts citrate to isocitrate?", "topics": []}, cfg)
        st = (await g.aget_state(cfg)).values
    assert (st["citations"][0]["page"], st["citations"][0]["kind"]) == (2, "figure")


@pytest.mark.asyncio
async def test_captioned_figure_clears_cached_study_guide(tmp_path, vision):
    result = await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    tid = result["topic_id"]
    topics._update_topic(tid, study_guide="cached before the figures existed", audio_script="old")
    await figures.caption_pending()
    t = topics.load_topics()[tid]
    assert t["study_guide"] is None and t["audio_script"] is None


@pytest.mark.asyncio
async def test_ingest_finds_tables_once_per_page(tmp_path, monkeypatch):
    import pymupdf

    calls = []
    real = pymupdf.Page.find_tables

    def counting(self, *a, **kw):
        calls.append(self.number)
        return real(self, *a, **kw)

    monkeypatch.setattr(pymupdf.Page, "find_tables", counting)
    result = await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert result["figures_queued"] == 2
    assert sorted(calls) == [0, 1]  # both pages have drawings; each is searched once


@pytest.mark.asyncio
async def test_source_deleted_during_upload_queues_no_figures(tmp_path, vision, monkeypatch):
    from core import ingest

    real = ingest.extract_figures

    def delete_then_extract(path, tables=None):
        topics.delete_source(topics.list_sources()[0]["id"])  # the user deletes the PDF mid-upload
        return real(path, tables)

    monkeypatch.setattr(ingest, "extract_figures", delete_then_extract)
    await ingest_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    assert _pending() == []
    await figures.caption_pending()
    assert _chunks() == []
