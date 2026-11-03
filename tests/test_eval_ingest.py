"""StudyMind — eval corpus ingest is idempotent and lands in the ods topic."""

import pytest

from core import topics
from core.config import settings
from eval import ingest as eval_ingest
from tests.pdfs import figure_pdf


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
