"""StudyMind — eval corpus ingest is idempotent and lands in the ods topic."""

import pytest

from core import topics
from core.config import settings
from eval import ingest as eval_ingest
from tests.pdfs import figure_pdf


def test_fetch_skips_the_download_when_chapters_exist(tmp_path, monkeypatch):
    for stem in eval_ingest.CHAPTERS:
        (tmp_path / f"{stem}.pdf").write_bytes(b"%PDF-1.4")

    def no_network(*a, **k):
        raise AssertionError("downloaded the book although every chapter is committed")

    monkeypatch.setattr(eval_ingest.httpx, "stream", no_network)
    assert len(eval_ingest.fetch(tmp_path)) == len(eval_ingest.CHAPTERS)


@pytest.mark.asyncio
async def test_build_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chunker", settings.chunker)
    monkeypatch.setattr(eval_ingest, "open_corpus", lambda corpus: setattr(settings, "chunker", corpus))
    pdf_dir = tmp_path / "corpus"
    pdf_dir.mkdir()
    figure_pdf(pdf_dir / "chapter.pdf")

    stats = await eval_ingest.build("structure", pdf_dir)
    again = await eval_ingest.build("structure", pdf_dir)

    assert stats["sources"] == 1 and again["sources"] == 1 and stats["chunks"] == again["chunks"] > 0
    assert [s["title"] for s in topics.list_sources("ods")] == ["chapter"]
