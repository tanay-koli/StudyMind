"""
StudyMind — Structured parsing into blocks that keep page and section heading.
"""

from dataclasses import dataclass
from statistics import median
from typing import Optional

import pymupdf

HEADING_SIZE_RATIO = 1.2
HEADING_MAX_WORDS = 12
_BOLD = 16  # PyMuPDF span flag


@dataclass
class Block:
    page: Optional[int]
    section_heading: Optional[str]
    text: str
    kind: str = "text"  # text | table
    start_s: Optional[float] = None  # video: where this text starts


def _table_markdown(rows: list[list]) -> str:
    """Markdown from extracted cells; all-empty rows and columns (ruling artefacts) are dropped."""
    cells = [[" ".join((c or "").split()).replace("|", "\\|") for c in row] for row in rows]
    cells = [row for row in cells if any(row)]
    if not cells:
        return ""
    keep = [i for i in range(len(cells[0])) if any(i < len(row) and row[i] for row in cells)]
    cells = [[row[i] if i < len(row) else "" for i in keep] for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + " --- |" * len(keep)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells[1:]])


def _tables(page) -> list[tuple[pymupdf.Rect, str]]:
    """(box, markdown) per table on the page, top to bottom."""
    # find_tables costs ~0.3 s a page; pages without vector drawings (plain prose) are skipped.
    # shortcut: a borderless table on a page with no drawing at all is missed, upgrade if real notes lose tables
    if not page.get_drawings():
        return []
    out = []
    for t in page.find_tables().tables:
        md = _table_markdown(t.extract())
        if md:
            out.append((pymupdf.Rect(t.bbox), md))
    return sorted(out, key=lambda x: x[0].y0)


def _lines(doc, tables: dict[int, list]):
    """Yield (page, block_index, text, size, bold, top) for every non-empty line outside a table."""
    for page in doc:
        boxes = [box for box, _ in tables.get(page.number + 1, [])]
        for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                rect = pymupdf.Rect(line["bbox"])
                if any(box.contains((rect.tl + rect.br) / 2) for box in boxes):
                    continue  # the table block carries it, as Markdown
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & _BOLD for s in spans)
                yield page.number + 1, b_idx, text, size, bold, rect.y0


def parse_pdf(path: str) -> list[Block]:
    with pymupdf.open(path) as doc:
        tables = {page.number + 1: t for page in doc if (t := _tables(page))}
        lines = list(_lines(doc, tables))
    pending = [(p, box.y0, md) for p, ts in sorted(tables.items()) for box, md in ts]
    if not lines:
        return [Block(p, None, md, kind="table") for p, _, md in pending]

    # Body size = median font size weighted by characters
    body = median([size for _, _, text, size, _, _ in lines for _ in text])

    blocks: list[Block] = []
    heading: Optional[str] = None
    current: Optional[tuple[int, int]] = None
    after_heading = False
    for page, b_idx, text, size, bold, top in lines:
        # Tables go in reading order: before the first line below them on their page
        while pending and (pending[0][0], pending[0][1]) <= (page, top):
            blocks.append(Block(pending[0][0], heading, pending.pop(0)[2], kind="table"))
            current = None
        if size >= HEADING_SIZE_RATIO * body or (bold and len(text.split()) <= HEADING_MAX_WORDS):
            # Consecutive heading lines (title + bold key term) stay together
            heading = f"{heading}\n{text}" if after_heading else text
            after_heading = True
            current = None
            continue
        after_heading = False
        if current == (page, b_idx):
            blocks[-1].text += "\n" + text
        else:
            blocks.append(Block(page, heading, text))
            current = (page, b_idx)
    blocks += [Block(p, heading, md, kind="table") for p, _, md in pending]
    if not any(b.kind == "text" for b in blocks):
        # Every line looked like a heading (e.g. all-bold slides): keep the text per page
        pages: dict[int, list[str]] = {}
        for page, _, text, _, _, _ in lines:
            pages.setdefault(page, []).append(text)
        blocks += [Block(p, None, "\n".join(t)) for p, t in pages.items()]
    return blocks


def text_blocks(text: str) -> list[Block]:
    """Non-PDF sources become one block without page or heading."""
    return [Block(None, None, text)] if text and text.strip() else []
