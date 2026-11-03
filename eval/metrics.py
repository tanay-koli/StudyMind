"""
StudyMind — Eval metrics. Gold is anchored by quote (or figure-on-page), not chunk id,
so the same dataset scores both chunkers and survives re-ingest.

A hit is a dict with source_title, page, kind, text; a gold item is
{source, page, quote} or {source, page, kind: "figure"}.
"""

import re
from collections import Counter

REFUSAL = "This isn't in your material"


def normalise(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text.lower())
    return " ".join(text.split())


def matches(hit: dict, gold: dict) -> bool:
    if hit["source_title"] != gold["source"]:
        return False
    if gold.get("kind") == "figure":
        return hit.get("kind") == "figure" and hit.get("page") == gold["page"]
    return normalise(gold["quote"]) in normalise(hit["text"])


def recall_at(hits: list[dict], gold: list[dict], k: int = 5) -> float:
    if not gold:
        return 0.0
    top = hits[:k]
    return sum(any(matches(h, g) for h in top) for g in gold) / len(gold)


def mrr(hits: list[dict], gold: list[dict]) -> float:
    for rank, h in enumerate(hits, start=1):
        if any(matches(h, g) for g in gold):
            return 1.0 / rank
    return 0.0


def citation_accuracy(answer: str, hits: list[dict], gold: list[dict]) -> float | None:
    """Share of [n] markers whose (1-based) hit matches gold; None if the answer cites nothing."""
    marks = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not marks:
        return None
    ok = sum(1 <= n <= len(hits) and any(matches(hits[n - 1], g) for g in gold) for n in marks)
    return ok / len(marks)


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in ca) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL)
