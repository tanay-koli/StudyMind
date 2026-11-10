"""StudyMind — eval report: per-config summary, README table, chart and marker section."""

import pytest

from eval import report

GOLD = [{"source": "ODS", "page": 1, "quote": "heaps are trees"}]
DATASET = {
    "q1": {"id": "q1", "type": "text", "gold": GOLD, "gold_answer": "trees"},
    "q2": {"id": "q2", "type": "text", "gold": GOLD, "gold_answer": "trees"},
    "q3": {"id": "q3", "type": "unanswerable", "gold": [], "gold_answer": "Not in the material."},
}
HIT = {"source_title": "ODS", "page": 1, "chunk_id": 1, "kind": "text", "score": 1.0, "text": "Heaps are trees."}


def _row(id, type, answer="Heaps are trees [1].", refused=False, error=None, judge=None):
    return {"id": id, "type": type, "question": "?", "hits": [HIT], "answer": answer, "refused": refused,
            "latency_s": 2.0, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "error": error,
            "judge": judge or {"faithfulness": 1.0, "correctness": 1.0}}


@pytest.fixture
def result_E():
    return {"config": {"name": "E"}, "rows": [
        _row("q1", "text"),
        _row("q2", "text", answer="", error="APIConnectionError: down", judge={"faithfulness": None, "correctness": None}),
        _row("q3", "unanswerable", answer="This isn't in your material.", refused=True,
             judge={"faithfulness": None, "correctness": None}),
    ]}


def test_summarise_excludes_errors_and_unanswerable_from_recall(result_E):
    s = report.summarise(result_E, DATASET)
    assert s["recall5"] == 1.0 and s["mrr"] == 1.0 and s["errors"] == 1 and s["refusal"] == 1.0
    assert s["faithfulness"] == 1.0 and s["correctness"] == 1.0 and s["latency"] == 2.0


def test_build_table_rows_and_chart(result_E):
    md = report.build({"E": result_E}, DATASET, labels=[])
    assert "| A | Cognee baseline (current app) | not measured (Cognee removed in Phase 1)" in md
    assert "| E (errors: 1) |" in md and "100.0" in md
    assert "judge unvalidated" in md and "```mermaid" in md and "xychart-beta" in md


def test_update_readme_appends_then_replaces_preserving_crlf(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(b"# StudyMind\r\n\r\nIntro.\r\n")
    report.update_readme(p, "one")
    report.update_readme(p, "two")
    text = p.read_bytes()
    assert text.startswith(b"# StudyMind\r\n\r\nIntro.\r\n") and text.count(b"<!-- eval:start -->") == 1
    assert b"two" in text and b"one" not in text


def test_stale_and_missing_rows_are_flagged_not_fatal(result_E):
    result_E["rows"] = [result_E["rows"][0], _row("q9", "text")]  # q9 was deleted from the dataset
    s = report.summarise(result_E, DATASET)
    assert s["n"] == 1 and s["stale"] == 1 and s["recall5"] == 1.0
    assert "| E (n=1/3, stale: 1) |" in report.build({"E": result_E}, DATASET, labels=[])


def test_unjudged_config_is_not_judge_failures(result_E):
    for r in result_E["rows"]:
        r.pop("judge")
    s = report.summarise(result_E, DATASET)
    assert s["judge_failures"] == 0 and s["unjudged"] == 2 and s["faithfulness"] is None
    md = report.build({"E": result_E}, DATASET, labels=[])
    assert "Not judged yet: E" in md and "Judge failures" not in md
