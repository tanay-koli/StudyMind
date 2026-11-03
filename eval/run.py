"""
StudyMind — Eval runner: answer every dataset question under one named config and record
hits, answer, refusal, latency and LLM usage. Metrics are computed later by eval.report.

Usage: python -m eval.run --config <name> [--limit N] [--dataset PATH]
Writes eval/results/<name>.json.
"""

import argparse
import asyncio
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core import retrieval, topics
from core.llm import llm
from core.query import _generate
from eval import configs
from eval.metrics import is_refusal

EVAL_DIR = Path(__file__).resolve().parent


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hit_record(h: dict) -> dict:
    return {"source_title": h["source_title"], "page": h["page"], "chunk_id": h["chunk_id"],
            "kind": (h.get("media") or {}).get("kind", "text"), "score": h["score"], "text": h["text"]}


async def _plain(row: dict) -> tuple[list[dict], str]:
    hits = await retrieval.get_retriever().retrieve(row["question"], row.get("topic_id"))
    answer = (await _generate(row["question"], hits))[0] if hits else ""
    return [dataclasses.asdict(h) for h in hits], answer


async def _agent(row: dict, name: str) -> tuple[list[dict], str]:
    from core import agent

    async with agent.open_graph() as g:
        cfg = {"configurable": {"thread_id": f"eval-{name}-{row['id']}-{uuid4().hex[:6]}"}}
        await g.ainvoke({"messages": [], "question": row["question"], "topics": [],
                         "topic_id": row.get("topic_id")}, cfg)
        state = (await g.aget_state(cfg)).values
    return state.get("documents") or [], state.get("answer", "")


async def _row(row: dict, name: str, cfg: dict) -> dict:
    before = dict(llm.usage)
    start = time.perf_counter()
    hits, answer, error = [], "", None
    try:
        hits, answer = await (_agent(row, name) if cfg["agent"] else _plain(row))
    except Exception as exc:  # one failed question must not lose the run
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": row["id"], "type": row.get("type"), "question": row["question"],
        "hits": [_hit_record(h) for h in hits], "answer": answer, "refused": is_refusal(answer),
        "latency_s": round(time.perf_counter() - start, 2),
        **{k: llm.usage[k] - before[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd")},
        "error": error,
    }


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=EVAL_DIR).stdout.strip()


async def run(name: str, dataset: Path, out_dir: Path, limit: int | None = None, corpus: bool = True) -> Path:
    cfg = configs.CONFIGS[name]
    if corpus:
        configs.open_corpus(cfg["corpus"])
    configs.apply(cfg)
    retrieval.ensure_indexed()

    rows = load_dataset(dataset)[:limit]
    titles = {s["title"] for s in topics.list_sources()}
    missing = sorted({g["source"] for r in rows for g in r.get("gold", [])} - titles)
    if missing:
        raise ValueError(f"gold sources not in the {cfg['corpus']} corpus: {missing}")

    results = [await _row(r, name, cfg) for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps({"config": {"name": name, **cfg}, "run_at": datetime.now().isoformat(timespec="seconds"),
                               "git_sha": _git_sha(), "rows": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=configs.ORDER)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    args = parser.parse_args()
    try:
        out = asyncio.run(run(args.config, args.dataset, EVAL_DIR / "results", args.limit))
    finally:
        retrieval.reset_store()  # close embedded Qdrant before interpreter shutdown
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
