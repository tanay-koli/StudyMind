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

    def test_build_connection_graph_with_overlap(self):
        """Should find shared entities between two result sets."""
        from core.graph_utils import build_connection_graph

        results_a = [
            {"text": "attention mechanism", "type": "entity", "source": None, "metadata": {}, "raw": None},
            {"text": "transformer", "type": "entity", "source": None, "metadata": {}, "raw": None},
        ]
        results_b = [
            {"text": "attention mechanism", "type": "entity", "source": None, "metadata": {}, "raw": None},
            {"text": "neural network", "type": "entity", "source": None, "metadata": {}, "raw": None},
        ]

        nodes, edges = build_connection_graph("transformers", "BERT", results_a, results_b)

        assert len(nodes) > 0, "Should have nodes"
        assert len(edges) > 0, "Should have edges"

        # Should have the two topic nodes
        labels = [n["label"] for n in nodes]
        assert "transformers" in labels
        assert "BERT" in labels

    def test_build_connection_graph_no_overlap(self):
        """Should return ([], []) when no shared entities exist.

        build_connection_graph intentionally returns empty results when
        there's no real overlap — fabricating a '?' bridge node was removed
        as it was misleading. Callers handle the empty state cleanly.
        """
        from core.graph_utils import build_connection_graph

        results_a = [
            {"text": "calculus", "type": "entity", "source": None, "metadata": {}, "raw": None},
        ]
        results_b = [
            {"text": "poetry", "type": "entity", "source": None, "metadata": {}, "raw": None},
        ]

        nodes, edges = build_connection_graph("math", "literature", results_a, results_b)

        # No overlap → clean empty state, not a fabricated bridge
        assert nodes == [], "No overlap should return empty nodes"
        assert edges == [], "No overlap should return empty edges"

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
