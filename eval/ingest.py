"""
StudyMind — Eval corpora: fetch the course PDFs and ingest them once per chunker into
data/eval/<fixed|structure>/, with figure captions. Re-running skips what is already there.

Usage: python -m eval.ingest [--corpus fixed|structure|both] [--no-caption]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pymupdf
from sqlmodel import func, select

from core import figures, retrieval, topics
from core.db import Chunk, ChunkMedia, PendingFigure, session
from core.ingest import ingest_pdf
from eval.configs import open_corpus

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
TOPIC = "ods"

# Open Data Structures (Pat Morin), Creative Commons Attribution; see eval/corpus/README.md
SOURCE_URL = "https://opendatastructures.org/ods-python.pdf"
# file stem -> 1-based inclusive page range in the full PDF
CHAPTERS: dict[str, tuple[int, int]] = {
    "ods-03-linked-lists": (73, 94),
    "ods-04-skiplists": (95, 112),
    "ods-06-binary-trees": (139, 156),
}


def fetch(dest: Path = CORPUS_DIR) -> list[Path]:
    """Download the book once and split out the chosen chapters (skips files already there)."""
    dest.mkdir(parents=True, exist_ok=True)
    book = dest / "ods-python.pdf"
    if not book.exists():
        with httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            book.write_bytes(r.read())
    out = []
    for stem, (first, last) in CHAPTERS.items():
        path = dest / f"{stem}.pdf"
        if not path.exists():
            with pymupdf.open(book) as src, pymupdf.open() as doc:
                doc.insert_pdf(src, from_page=first - 1, to_page=last - 1)
                doc.save(path)
        out.append(path)
    return out


async def build(corpus: str, pdf_dir: Path, topic: str = TOPIC, caption: bool = True) -> dict:
    """Ingest every chapter PDF not yet a source in the topic, then caption pending figures."""
    open_corpus(corpus)
    tid = topics.create_topic(topic)["id"]
    have = {s["title"] for s in topics.list_sources(tid)}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        if pdf.name == "ods-python.pdf" or pdf.stem in have:
            continue
        res = await ingest_pdf(str(pdf), title=pdf.stem, topic_id=tid)
        print(f"{corpus}: {pdf.stem} -> {res['status']} {res.get('error') or ''}")
    while caption and await figures.caption_pending():  # 0 when done or when the vision model is unreachable
        pass
    with session() as s:
        count = lambda q: s.exec(q).one()  # noqa: E731
        return {
            "sources": len(topics.list_sources(tid)),
            "chunks": count(select(func.count()).select_from(Chunk).where(Chunk.topic_id == tid)),
            "figures": count(select(func.count()).select_from(ChunkMedia).join(Chunk, Chunk.id == ChunkMedia.chunk_id)
                             .where(Chunk.topic_id == tid, ChunkMedia.kind == "figure")),
            "figures_pending": count(select(func.count()).select_from(PendingFigure)),
        }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["fixed", "structure", "both"], default="both")
    parser.add_argument("--no-caption", action="store_true", help="skip figure captioning (CI has no vision model)")
    args = parser.parse_args()
    fetch()
    for corpus in (["fixed", "structure"] if args.corpus == "both" else [args.corpus]):
        try:
            # Only the structure corpus runs with figures on (G, H), so only it is captioned
            print(corpus, asyncio.run(build(corpus, CORPUS_DIR, caption=corpus == "structure" and not args.no_caption)))
        finally:
            retrieval.reset_store()


if __name__ == "__main__":
    main()
