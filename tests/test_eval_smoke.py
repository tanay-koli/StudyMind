"""StudyMind — retrieval-only smoke check against the committed baseline."""

from eval import smoke

BASE = {"config": "E", "recall5": 0.7333, "n": 15, "source": "eval/results/E.json"}


def _case(hit_count, extra_unanswerable=False):
    """15 answerable rows, the first `hit_count` retrieve their gold."""
    dataset, rows = {}, []
    for i in range(15):
        gold = [{"source": "s", "page": 1, "quote": f"quote {i}"}]
        dataset[f"q{i}"] = {"id": f"q{i}", "type": "text", "gold": gold}
        text = f"quote {i}" if i < hit_count else "nothing"
        rows.append({"id": f"q{i}", "error": None,
                     "hits": [{"source_title": "s", "page": 1, "kind": "text", "text": text}]})
    if extra_unanswerable:
        dataset["u"] = {"id": "u", "type": "unanswerable", "gold": []}
        rows.append({"id": "u", "error": None, "hits": []})
    return {"rows": rows}, dataset


def test_smoke_passes_at_one_lost_question():
    result, dataset = _case(10)
    assert smoke.check(result, dataset, BASE)[0] is True


def test_smoke_fails_at_two_lost_questions():
    result, dataset = _case(9)
    assert smoke.check(result, dataset, BASE)[0] is False


def test_smoke_ignores_unanswerable_rows():
    result, dataset = _case(10, extra_unanswerable=True)
    ok, recall = smoke.check(result, dataset, BASE)
    assert ok and abs(recall - 10 / 15) < 1e-9
