"""StudyMind — eval configs and the --config runner."""

import json

import pytest

from core.config import settings
from eval import configs, run
from tests.planting import plant

SWITCHES = ("retriever", "rerank", "figures", "query_rewrite", "embed_model", "agent_db_path", "chunker")


@pytest.fixture
def keep_settings(monkeypatch, tmp_path):
    """Snapshot every setting a config can change, so the test restores it."""
    for k in SWITCHES:
        monkeypatch.setattr(settings, k, getattr(settings, k))
    monkeypatch.setattr(settings, "agent_db_path", str(tmp_path / "agent.db"))


def _dataset(tmp_path, source="ODS notes"):
    tid, _ = plant(["heaps support insert in log n time", "skiplists are layered lists"], {}, topic="ODS")
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({
        "id": "q1", "type": "text", "topic_id": tid, "question": "heaps support insert",
        "gold_answer": "log n", "gold": [{"source": source, "page": 1, "quote": "heaps support insert"}],
    }) + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_run_writes_config_json(tmp_path, keep_settings):
    out = await run.run("E", _dataset(tmp_path), tmp_path / "results", corpus=False)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert out.name == "E.json" and doc["config"]["name"] == "E" and doc["config"]["corpus"] == "structure"
    row = doc["rows"][0]
    assert row["hits"][0]["source_title"] == "ODS notes" and row["hits"][0]["kind"] == "text"
    assert {"answer", "refused", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd", "error"} <= row.keys()
    assert row["answer"] == "Transformer Models" and row["refused"] is False and row["error"] is None


@pytest.mark.asyncio
async def test_unknown_gold_source_fails_fast(tmp_path, keep_settings):
    with pytest.raises(ValueError, match="Missing Book"):
        await run.run("E", _dataset(tmp_path, source="Missing Book"), tmp_path / "results", corpus=False)


@pytest.mark.asyncio
async def test_agent_config_records_suffixed_refusal(tmp_path, keep_settings, monkeypatch):
    monkeypatch.setattr(settings, "relevance_min", 99.0)  # nothing relevant -> not_found node
    out = await run.run("H", _dataset(tmp_path), tmp_path / "results", corpus=False)
    row = json.loads(out.read_text(encoding="utf-8"))["rows"][0]
    assert row["refused"] is True and row["answer"].startswith("This isn't in your material. Searched")


def test_configs_ladder():
    C = configs.CONFIGS
    assert configs.ORDER == list(C)
    assert C["B"]["corpus"] == "fixed" and C["D"]["rerank"] is False
    assert C["G"]["figures"] is True and C["H"]["agent"] is True
    assert all(C[n]["figures"] is False for n in ("B", "C", "D", "E", "F-multi", "F-hyde", "E-nomic"))


def test_open_corpus_isolates_every_data_path(tmp_path, keep_settings, monkeypatch):
    from core import config

    monkeypatch.setattr(configs, "CORPUS_ROOT", tmp_path / "eval")
    monkeypatch.setattr(config, "FIGURE_DIR", config.FIGURE_DIR)
    configs.open_corpus("structure")
    root = (tmp_path / "eval" / "structure").as_posix()
    paths = [settings.database_url, settings.qdrant_path, settings.agent_db_path, config.FIGURE_DIR.as_posix()]
    assert all(root in p for p in paths) and settings.chunker == "structure"


def test_h_label_says_the_agent_replaces_query_rewriting():
    # the agent searches with get_retriever(rewrite=False), so H drops G's multi-query / HyDE
    assert "own query rewrite" in configs.CONFIGS["H"]["change"]
