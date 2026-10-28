"""StudyMind — eval runner writes one result line per dataset question."""

import json

import pytest

from core import topics
from eval import run


@pytest.mark.asyncio
async def test_run_writes_results(tmp_path):
    from core.ingest import ingest_text

    await ingest_text("Gradient descent steps against the gradient", title="Opt", topic_id="general")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "question": "gradient descent", "topic_id": "general",
        "expected_source": "Opt", "expected_pages": [], "reference_answer": "x",
    }) + "\n", encoding="utf-8")

    out = await run.run(dataset, tmp_path / "results", retriever="hybrid", embed="bge-small")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert out.name.startswith("hybrid-bge-small-")
    assert [(r["id"], r["hits"][0]["source_title"], r["error"]) for r in rows] == [("q1", "Opt", None)]
    assert {"page", "chunk_id", "score"} <= rows[0]["hits"][0].keys()
    assert rows[0]["answer"] == "Transformer Models"


@pytest.mark.asyncio
async def test_run_reindexes_empty_collection(tmp_path, monkeypatch):
    from core.ingest import ingest_text
    from core import retrieval

    await ingest_text("Gradient descent", title="Opt", topic_id="general")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"id": "q1", "question": "gradient", "topic_id": "general"}) + "\n", encoding="utf-8")

    await run.run(dataset, tmp_path / "results", retriever="hybrid", embed="nomic")
    assert retrieval.store().count("chunks_nomic").count == 1
    assert topics.list_sources()  # source untouched
