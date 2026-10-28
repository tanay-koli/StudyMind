"""
StudyMind — Eval runner: answer every dataset question with one retriever and
record the hits, so retrievers (and embedding models) can be compared later.

Usage: python -m eval.run --retriever hybrid|dense [--embed bge-small|nomic] [--rewrite none|multi|hyde]
Writes eval/results/<retriever>-<embed>-<rewrite>-<YYYY-MM-DD>.jsonl. Metrics come in Phase 6.
"""

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from core import retrieval
from core.config import settings
from core.query import _answer
from core.topics import init_db

EVAL_DIR = Path(__file__).resolve().parent


async def run(dataset: Path, out_dir: Path, retriever: str, embed: str, rewrite: str = "none") -> Path:
    settings.retriever = retriever
    settings.embed_model = embed
    settings.query_rewrite = rewrite
    init_db()
    retrieval.ensure_indexed()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{retriever}-{embed}-{rewrite}-{date.today().isoformat()}.jsonl"
    with dataset.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            res = await _answer(row["question"], row.get("topic_id"), "No results found.")
            hits = [{k: c[k] for k in ("source_title", "page", "chunk_id", "score")} for c in res["citations"]]
            w.write(json.dumps({
                "id": row["id"], "question": row["question"], "topic_id": row.get("topic_id"),
                "hits": hits, "answer": res["answer"], "error": res["error"],
            }, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--embed", choices=list(retrieval.EMBED_MODELS), default=settings.embed_model)
    parser.add_argument("--rewrite", choices=["none", "multi", "hyde"], default=settings.query_rewrite)
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    args = parser.parse_args()
    try:
        out = asyncio.run(run(args.dataset, EVAL_DIR / "results", args.retriever, args.embed, args.rewrite))
    finally:
        retrieval.reset_store()  # close embedded Qdrant before interpreter shutdown
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
