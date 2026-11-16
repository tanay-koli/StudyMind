"""
StudyMind — Retrieval-only smoke check: fail when Recall@5 drops more than one question
below the committed baseline.

Usage: python -m eval.smoke RESULT_JSON
"""

import json
import sys
from pathlib import Path

from eval.metrics import recall_at
from eval.run import EVAL_DIR, load_dataset

BASELINE = EVAL_DIR / "smoke_baseline.json"


def score(result: dict, dataset: dict[str, dict]) -> tuple[float, int, int]:
    """(recall5, answerable rows present, rows with an error); errored rows score 0."""
    rows = [r for r in result["rows"] if r["id"] in dataset and dataset[r["id"]]["type"] != "unanswerable"]
    hit = sum(recall_at(r["hits"], dataset[r["id"]]["gold"]) for r in rows if not r["error"])
    recall = hit / len(rows) if rows else 0.0
    return recall, len(rows), sum(bool(r["error"]) for r in result["rows"])


def check(result: dict, dataset: dict[str, dict], baseline: dict) -> tuple[bool, float]:
    """(passed, recall5): fails on any errored row or a row count that differs from the baseline's."""
    recall, n, errors = score(result, dataset)
    # shortcut: one question of tolerance, fine at n=15; switch to a 5-point drop once the set reaches ~150 questions.
    return errors == 0 and n == baseline["n"] and recall >= baseline["recall5"] - 1.5 / n, recall


def main() -> None:
    result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dataset = {r["id"]: r for r in load_dataset(EVAL_DIR / "dataset.jsonl")}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ok, recall = check(result, dataset, baseline)
    _, n, errors = score(result, dataset)
    floor = baseline["recall5"] - 1.5 / baseline["n"]
    print(f"Recall@5 {recall * 100:.1f}% over {n} rows, {errors} errors (baseline {baseline['recall5'] * 100:.1f}%, "
          f"floor {floor * 100:.1f}%) {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
