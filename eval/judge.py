"""
StudyMind — Eval judge: faithfulness (share of answer claims the retrieved passages support) and
correctness (answer vs gold answer), plus a hand-label calibration that reports agreement.

Usage: python -m eval.judge <config>        score eval/results/<config>.json in place
       python -m eval.judge --calibrate     hand-label 30 answers, then print agreement
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

from core.config import settings
from core.llm import llm
from eval.metrics import cohen_kappa

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
CACHE = RESULTS / "judge_cache.json"
LABELS = EVAL_DIR / "judge_labels.jsonl"
PROMPT_VERSION = 1

FAITH_SYSTEM = ("You check whether each claim in an answer is supported by the numbered passages. "
                "Split the answer into atomic factual claims (skip citations and filler).")
CORRECT_SYSTEM = ("You compare a student-assistant answer with a reference answer. correct = same key facts; "
                  "partial = some key facts or minor errors; wrong = missing or contradicting the key facts.")
VERDICTS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
LETTER = {1.0: "c", 0.5: "p", 0.0: "w"}


def _model() -> str:
    return settings.judge_model or settings.llm_model


async def _ask(prompt: str, system: str, cache: dict | None = None) -> dict | None:
    """JSON reply from the judge; one retry, then None. Successful replies are cached by model + prompt."""
    key = hashlib.sha256(json.dumps([_model(), PROMPT_VERSION, system, prompt]).encode()).hexdigest()
    if cache is not None and key in cache:
        return cache[key]
    for _ in range(2):
        try:
            reply = await llm.complete_json(prompt, system, model=_model())
        except ValueError:
            continue
        if cache is not None:
            cache[key] = reply
        return reply
    return None


def context_text(hits: list[dict]) -> str:
    return "\n\n".join(f"[{n}] {h['source_title']}, p.{h['page']}\n{h['text']}" for n, h in enumerate(hits, 1))


async def faithfulness(context: str, answer: str, cache: dict | None = None) -> float | None:
    reply = await _ask(
        f"{context}\n\nAnswer:\n{answer}\n\n"
        'Reply with JSON: {"claims": [{"claim": "...", "supported": true or false}]}', FAITH_SYSTEM, cache)
    claims = (reply or {}).get("claims")
    if not isinstance(claims, list):
        return None
    return sum(bool(c.get("supported")) for c in claims if isinstance(c, dict)) / len(claims) if claims else 1.0


async def correctness(question: str, gold_answer: str, answer: str, cache: dict | None = None) -> float | None:
    reply = await _ask(
        f"Question: {question}\n\nReference answer:\n{gold_answer}\n\nAnswer to grade:\n{answer}\n\n"
        'Reply with JSON: {"verdict": "correct" | "partial" | "wrong"}', CORRECT_SYSTEM, cache)
    return VERDICTS.get(str((reply or {}).get("verdict", "")).lower())


async def judge_file(result_path: Path, dataset: dict[str, dict], cache_path: Path = CACHE) -> dict:
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    judged = failures = 0
    try:  # a crash (Ctrl-C, Ollama down) keeps every verdict already paid for
        for row in doc["rows"]:
            scores = {"faithfulness": None, "correctness": None}
            tried = []
            if not row["error"] and not row["refused"]:
                tried.append("faithfulness")
                scores["faithfulness"] = await faithfulness(context_text(row["hits"]), row["answer"], cache)
            if not row["error"] and row["type"] != "unanswerable":
                tried.append("correctness")
                gold = dataset[row["id"]]["gold_answer"]
                scores["correctness"] = await correctness(row["question"], gold, row["answer"], cache)
            if tried:
                judged += 1
                failures += any(scores[m] is None for m in tried)
            row["judge"] = scores
    finally:
        result_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {"judged": judged, "failures": failures}


def load_results(results_dir: Path = RESULTS) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(results_dir.glob("*.json")) if p != results_dir / CACHE.name}


def load_labels(path: Path = LABELS) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def sample(results: dict[str, dict], labelled: set[tuple[str, str]], n: int = 30) -> list[tuple[str, dict]]:
    """Up to n judged rows, round-robin over (type, judge verdict) buckets so easy cases don't dominate."""
    buckets: dict[tuple, list] = {}
    for name, doc in sorted(results.items()):
        for row in doc["rows"]:
            j = row.get("judge") or {}
            if (name, row["id"]) in labelled or j.get("correctness") is None or j.get("faithfulness") is None:
                continue
            buckets.setdefault((row["type"], j["correctness"]), []).append((name, row))
    rng = random.Random(0)
    queues = [rng.sample(v, len(v)) for _, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))]
    picked = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    return picked


def _choice(ask, prompt: str, options: str) -> str:
    while (reply := ask(prompt).strip().lower()) not in options:
        pass
    return reply


def calibrate(results: dict[str, dict], dataset: dict[str, dict], labels_path: Path = LABELS,
              n: int = 30, ask=input) -> None:
    """Ask for your verdict on judged answers; each label is saved as soon as it is given."""
    done = {(x["config"], x["id"]) for x in load_labels(labels_path)}
    todo = sample(results, done, n - len(done))
    for i, (name, row) in enumerate(todo, 1):
        print(f"\n=== {i}/{len(todo)} · {row['id']} ({row['type']}) ===\nQ: {row['question']}\n"
              f"Gold: {dataset[row['id']]['gold_answer']}\n\nAnswer:\n{row['answer']}\n\n"
              f"Context:\n{context_text(row['hits'])}\n")
        correct = _choice(ask, "Correct? [c/p/w] ", ("c", "p", "w"))
        faithful = _choice(ask, "All claims supported by the context? [y/n] ", ("y", "n")) == "y"
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"config": name, "id": row["id"], "correct": correct, "faithful": faithful}) + "\n")


def agreement(labels: list[dict], results: dict[str, dict]) -> dict:
    rows = {(name, r["id"]): r.get("judge") or {} for name, doc in results.items() for r in doc["rows"]}
    pairs = [(x, rows[(x["config"], x["id"])]) for x in labels if (x["config"], x["id"]) in rows]
    pairs = [(x, j) for x, j in pairs if j.get("correctness") is not None and j.get("faithfulness") is not None]
    out = {"n": len(pairs)}
    for metric, mine, theirs in (
        ("correctness", [x["correct"] for x, _ in pairs], [LETTER[j["correctness"]] for _, j in pairs]),
        ("faithfulness", [x["faithful"] for x, _ in pairs], [j["faithfulness"] == 1 for _, j in pairs]),
    ):
        out[metric] = ({"agree": sum(a == b for a, b in zip(mine, theirs)) / len(pairs),
                        "kappa": cohen_kappa(mine, theirs)} if pairs else None)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from eval.run import load_dataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    args = parser.parse_args()
    dataset = {r["id"]: r for r in load_dataset(args.dataset)}
    if args.calibrate:
        results = load_results()
        try:
            calibrate(results, dataset)
        except (KeyboardInterrupt, EOFError):
            print("\nStopped; labels so far are saved.")
        print(json.dumps(agreement(load_labels(), results), indent=1))
    elif args.config:
        print(asyncio.run(judge_file(RESULTS / f"{args.config}.json", dataset)))
    else:
        parser.error("give a config name or --calibrate")


if __name__ == "__main__":
    main()
