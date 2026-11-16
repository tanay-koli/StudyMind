"""
StudyMind — Ingestion tests (offline: models are faked in conftest.py)

Run with: pytest -v
"""

import pytest


class TestTextIngestion:
    """Test ingesting plain text notes."""

    @pytest.mark.asyncio
    async def test_ingest_text_returns_status(self):
        """ingest_text() should return a structured status dict."""
        from core.ingest import ingest_text

        result = await ingest_text(
            text="The transformer architecture uses self-attention mechanisms.",
            title="Test Note",
        )

        assert isinstance(result, dict)
        assert "status" in result
        assert "title" in result
        assert "dataset_name" not in result and "cognee" not in result
        assert result["title"] == "Test Note"
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_ingest_text_records_source(self):
        """A successful ingestion is stored as a source row on its topic."""
        from core.ingest import ingest_text
        from core.topics import create_topic, list_sources

        tid = create_topic("Transformers")["id"]
        result = await ingest_text(text="Attention is all you need.", title="Paper", topic_id=tid)

        assert [s["id"] for s in list_sources(tid)] == [result["source_id"]]

    @pytest.mark.asyncio
    async def test_ingest_is_local_only(self, offline):
        """Ingest never calls a remote service."""
        from core.ingest import ingest_text

        result = await ingest_text(text="hello", title="Note")
        assert result["status"] == "success" and result["chunk_count"] == 1
        assert offline == []

    @pytest.mark.asyncio
    async def test_index_failure_rolls_back(self, monkeypatch):
        """If indexing fails, no source, chunk rows or points are left behind."""
        from sqlmodel import select
        from core import retrieval
        from core.db import Chunk, session
        from core.ingest import ingest_text
        from core.topics import list_sources

        def boom(texts, *, query):
            raise RuntimeError("embedder down")

        monkeypatch.setattr(retrieval, "embed_dense", boom)
        result = await ingest_text(text="hello", title="Note")
        assert result["status"] == "error" and "embedder down" in result["error"]
        assert list_sources() == []
        with session() as s:
            assert s.exec(select(Chunk)).all() == []

    @pytest.mark.asyncio
    async def test_ingest_empty_text_returns_error(self):
        """Ingesting empty text should return an error status."""
        from core.ingest import ingest_text

        result = await ingest_text(text="", title="Empty")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()


class TestPDFIngestion:
    """Test PDF parsing and ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_pdf_missing_file(self):
        """Ingesting a non-existent PDF should return an error."""
        from core.ingest import ingest_pdf

        result = await ingest_pdf("/nonexistent/path/test.pdf")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_ingest_pdf_chunks_keep_pages(self, tmp_path):
        """Every chunk of a PDF keeps the page it came from."""
        import pymupdf
        from sqlmodel import select
        from core.db import Chunk, session
        from core.ingest import ingest_pdf

        doc = pymupdf.open()
        for i in (1, 2):
            doc.new_page().insert_text((72, 72), f"Content of page {i}")
        path = tmp_path / "two.pdf"
        doc.save(path)
        doc.close()

        result = await ingest_pdf(str(path))
        assert result["status"] == "success" and result["chunk_count"] == 2
        with session() as s:
            pages = s.exec(select(Chunk.page).where(Chunk.source_id == result["source_id"])).all()
        assert sorted(pages) == [1, 2]


class TestGraphUtils:
    """Test graph utility functions."""

    def test_empty_graph_html(self):
        """Empty graph should produce valid HTML."""
        from core.graph_utils import render_pyvis_graph

        html = render_pyvis_graph([], [], "a", "b")
        assert "<div" in html
        assert "No connections" in html

    def test_source_color_mapping(self):
        """Source colors should be defined for all types."""
        from core.graph_utils import SOURCE_COLORS

        assert "pdf" in SOURCE_COLORS
        assert "youtube" in SOURCE_COLORS
        assert "web" in SOURCE_COLORS
        assert "note" in SOURCE_COLORS
        assert "topic" in SOURCE_COLORS


class TestYouTubeTimestamps:
    @staticmethod
    def _fake_api(monkeypatch, snippets):
        import youtube_transcript_api

        class FakeApi:
            def fetch(self, video_id):
                return snippets

        monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)

    @staticmethod
    def _media():
        from sqlmodel import select
        from core.db import Chunk, ChunkMedia, session

        with session() as s:
            return s.exec(select(Chunk.text, ChunkMedia).join(ChunkMedia, ChunkMedia.chunk_id == Chunk.id, isouter=True)
                          .order_by(Chunk.id)).all()

    @pytest.mark.asyncio
    async def test_youtube_chunks_cite_start_time(self, monkeypatch):
        from types import SimpleNamespace
        from core import chunking
        from core.ingest import ingest_youtube

        snippets = [SimpleNamespace(text="alpha beta", start=0.0, duration=3),
                    SimpleNamespace(text="gamma", start=12.5, duration=3),
                    SimpleNamespace(text="delta epsilon", start=75.2, duration=3)]
        self._fake_api(monkeypatch, snippets)
        result = await ingest_youtube("https://youtu.be/abc123")
        assert result["status"] == "success"
        [(text, m)] = self._media()
        assert (m.kind, m.start_s, m.url) == ("video", 0.0, "https://youtu.be/abc123?t=0")

        from core import ingest
        monkeypatch.setattr(ingest, "chunk", lambda blocks: chunking.structure(blocks, max_tokens=2))
        await ingest_youtube("https://youtu.be/xyz789")
        starts = [(m.start_s, m.url) for _, m in self._media()[1:]]
        assert starts == [(0.0, "https://youtu.be/xyz789?t=0"), (12.5, "https://youtu.be/xyz789?t=12"),
                          (75.2, "https://youtu.be/xyz789?t=75")]

    @pytest.mark.asyncio
    async def test_youtube_without_start_times_still_ingests(self, monkeypatch):
        from core.ingest import ingest_youtube

        self._fake_api(monkeypatch, [{"text": "old api snippet one"}, {"text": "and two"}])
        result = await ingest_youtube("https://youtu.be/old111")
        assert result["status"] == "success" and result["chunk_count"] >= 1
        assert all(m is None for _, m in self._media())


    @pytest.mark.asyncio
    async def test_youtube_dict_snippets_keep_start_time(self, monkeypatch):
        from core.ingest import ingest_youtube

        self._fake_api(monkeypatch, [{"text": "old api snippet", "start": 42.0}])
        await ingest_youtube("https://youtu.be/dict42")
        [(_, m)] = self._media()
        assert (m.kind, m.start_s, m.url) == ("video", 42.0, "https://youtu.be/dict42?t=42")
