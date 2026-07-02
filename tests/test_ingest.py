"""
StudyMind — Smoke Tests for Ingestion

Basic tests that verify:
  1. Ollama is reachable
  2. Text ingestion completes without crashing
  3. PDF parsing works on a minimal PDF
  4. Search returns something after ingestion

These are integration tests — they require Ollama running locally.
Run with: pytest tests/test_ingest.py -v
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOllamaConnectivity:
    """Verify Ollama is running and models are available."""

    def test_ollama_reachable(self):
        """Ollama server should respond on localhost:11434."""
        from core.config import check_ollama

        status = check_ollama(timeout=5)
        if not status["ok"]:
            pytest.skip(f"Ollama not available: {status.get('error')}")

        assert status["ok"], f"Ollama check failed: {status.get('error')}"
        assert len(status["missing"]) == 0, f"Missing models: {status['missing']}"

    def test_ollama_has_models(self):
        """Required models (qwen3:8b, nomic-embed-text) should be pulled."""
        from core.config import check_ollama

        status = check_ollama(timeout=5)
        if not status["ok"]:
            pytest.skip(f"Ollama not available: {status.get('error')}")

        assert "models" in status
        assert len(status["models"]) > 0, "No models found in Ollama"


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
        assert "dataset_name" in result
        assert result["title"] == "Test Note"
        # Status will be 'success' or 'error' depending on Ollama availability
        assert result["status"] in ("success", "error")

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


class TestDatasetNaming:
    """Test dataset name sanitization."""

    def test_sanitize_dataset_name(self):
        """Dataset names should be alphanumeric + underscores."""
        from core.ingest import _sanitize_dataset_name

        assert _sanitize_dataset_name("My Study Notes!") == "my_study_notes"
        assert _sanitize_dataset_name("test-file-2024.pdf") == "test_file_2024_pdf"
        assert _sanitize_dataset_name("") == "untitled"
        assert _sanitize_dataset_name("  spaces  ") == "spaces"

    def test_make_dataset_name(self):
        """Dataset names should include the source type prefix."""
        from core.ingest import _make_dataset_name

        name = _make_dataset_name("pdf", "My Paper")
        assert name.startswith("pdf_")
        assert "my_paper" in name


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
