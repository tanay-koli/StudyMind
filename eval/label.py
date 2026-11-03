"""
StudyMind — Eval labelling CLI (runs on the structure corpus).

  python -m eval.label propose --n 8 --type text|diagram
      The LLM drafts a question, gold answer and verbatim quote from an unused chunk;
      you accept, edit or skip each.
  python -m eval.label find "your question" --type text|multihop|cross|unanswerable
      Shows the top 10 hits; tick the gold ones by number (none = unanswerable).

Accepted rows are appended to eval/dataset.jsonl straight away.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from sqlmodel import select

from core import retrieval
from core.db import Chunk, ChunkMedia, Source, session
from core.llm import llm
from eval.metrics import normalise

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
MAX_QUOTE_WORDS = 40
UNANSWERABLE = "Not in the material."

DRAFT_SYSTEM = "You write exam-style study questions about one passage of a textbook."


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_id(rows: list[dict]) -> str:
    return f"q{max((int(r['id'][1:]) for r in rows), default=0) + 1:03d}"


def first_quote(text: str) -> str | None:
    """A verbatim snippet of 5-40 words: the first such sentence, else the first 40 words."""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if 5 <= len(sentence.split()) <= MAX_QUOTE_WORDS:
            return sentence
    words = text.split()
    return " ".join(words[:MAX_QUOTE_WORDS]) if len(words) >= 5 else None


def _verbatim(quote: str, text: str) -> bool:
    return 0 < len(quote.split()) <= MAX_QUOTE_WORDS and normalise(quote) in normalise(text)


async def draft(chunk_text: str, kind: str) -> dict | None:
    """{question, gold_answer, quote} for a chunk; for text the quote must be verbatim (one re-ask)."""
    what = ("the figure this caption describes" if kind == "diagram" else "this passage")
    prompt = (f"Passage:\n{chunk_text}\n\nWrite one question a student could answer only from {what}, "
              "its short reference answer, and the exact sentence fragment (copied verbatim, at most "
              f"{MAX_QUOTE_WORDS} words) that answers it.\n"
              'Reply with JSON: {"question": "...", "gold_answer": "...", "quote": "..."}')
    for _ in range(2):
        try:
            d = await llm.complete_json(prompt, DRAFT_SYSTEM)
        except ValueError:
            continue
        if not all(isinstance(d.get(k), str) and d[k].strip() for k in ("question", "gold_answer")):
            continue
        if kind == "diagram" or _verbatim(str(d.get("quote", "")), chunk_text):
            return {k: str(d.get(k, "")).strip() for k in ("question", "gold_answer", "quote")}
    return None


def _candidates(qtype: str, used: set[tuple[str, int]]) -> list[tuple[Chunk, str]]:
    """Unused chunks (figures for diagram questions), at most one per (source, page), in a fixed shuffle."""
    with session() as s:
        rows = s.exec(select(Chunk, Source.title, ChunkMedia.kind)
                      .join(Source, Chunk.source_id == Source.id)
                      .join(ChunkMedia, ChunkMedia.chunk_id == Chunk.id, isouter=True)
                      .order_by(Chunk.id)).all()
    rows = [(c, t) for c, t, k in rows if (k == "figure") == (qtype == "diagram") and k != "video"]
    random.Random(0).shuffle(rows)
    seen, out = set(used), []
    for c, title in rows:
        if (title, c.page) not in seen:
            seen.add((title, c.page))
            out.append((c, title))
    return out


def _edit(d: dict, ask) -> dict:
    return {k: ask(f"{k} [{v}]: ").strip() or v for k, v in d.items()}


async def propose(n: int, qtype: str, dataset_path: Path = DATASET, ask=input) -> int:
    used = {(g["source"], g["page"]) for r in _load(dataset_path) for g in r.get("gold", [])}
    accepted = 0
    for chunk, title in _candidates(qtype, used):
        if accepted >= n:
            break
        d = await draft(chunk.text, qtype)
        if d is None:
            print(f"(no usable draft for {title} p.{chunk.page}; skipped)")
            continue
        print(f"\n--- {title} p.{chunk.page} ---\n{chunk.text}\n\nQ: {d['question']}\nA: {d['gold_answer']}"
              + (f"\nQuote: {d['quote']}" if qtype != "diagram" else ""))
        choice = ""
        while choice not in ("a", "e", "s", "q"):
            choice = ask("[a]ccept / [e]dit / [s]kip / [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "e":
            d = _edit(d, ask)
            if qtype != "diagram" and not _verbatim(d["quote"], chunk.text):
                print("Quote is not verbatim in the passage; skipped.")
                continue
        gold = ({"source": title, "page": chunk.page, "kind": "figure"} if qtype == "diagram"
                else {"source": title, "page": chunk.page, "quote": d["quote"]})
        _append(dataset_path, {"id": next_id(_load(dataset_path)), "type": qtype, "topic_id": chunk.topic_id,
                               "question": d["question"], "gold_answer": d["gold_answer"], "gold": [gold]})
        accepted += 1
    return accepted


async def find(question: str, qtype: str, dataset_path: Path = DATASET, ask=input, topic_id: str = "ods") -> dict:
    hits = await retrieval.get_retriever(rewrite=False).retrieve(question, topic_id, k=10)
    for i, h in enumerate(hits, 1):
        kind = (h.media or {}).get("kind", "text")
        print(f"\n[{i}] {h.source_title} p.{h.page} ({kind})\n{h.text[:400]}")
    picks = [int(x) for x in ask("Gold hit numbers (space-separated, empty = none): ").split()
             if x.isdigit() and 1 <= int(x) <= len(hits)]
    gold = []
    for h in (hits[i - 1] for i in picks):
        if (h.media or {}).get("kind") == "figure":
            gold.append({"source": h.source_title, "page": h.page, "kind": "figure"})
        elif quote := first_quote(h.text):
            gold.append({"source": h.source_title, "page": h.page, "quote": quote})
    default = UNANSWERABLE if not gold else ""
    answer = ask(f"Gold answer{f' [{default}]' if default else ''}: ").strip() or default
    row = {"id": next_id(_load(dataset_path)), "type": qtype, "topic_id": topic_id, "question": question,
           "gold_answer": answer, "gold": gold}
    _append(dataset_path, row)
    return row


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from eval.configs import open_corpus

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--type", choices=["text", "diagram"], default="text")
    f = sub.add_parser("find")
    f.add_argument("question")
    f.add_argument("--type", choices=["text", "multihop", "cross", "unanswerable"], default="text")
    f.add_argument("--topic", default="ods")
    args = parser.parse_args()
    open_corpus("structure")
    try:
        if args.cmd == "propose":
            print(f"Accepted {asyncio.run(propose(args.n, args.type))}")
        else:
            print(json.dumps(asyncio.run(find(args.question, args.type, topic_id=args.topic)), ensure_ascii=False))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; accepted rows are saved.")
    finally:
        retrieval.reset_store()


if __name__ == "__main__":
    main()
