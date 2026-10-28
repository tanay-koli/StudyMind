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


def _lines(doc):
    """Yield (page, block_index, text, size, bold) for every non-empty line."""
    for page in doc:
        for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & _BOLD for s in spans)
                yield page.number + 1, b_idx, text, size, bold


def parse_pdf(path: str) -> list[Block]:
    with pymupdf.open(path) as doc:
        lines = list(_lines(doc))
    if not lines:
        return []

    # Body size = median font size weighted by characters
    body = median([size for _, _, text, size, _ in lines for _ in text])

    blocks: list[Block] = []
    heading: Optional[str] = None
    current: Optional[tuple[int, int]] = None
    after_heading = False
    for page, b_idx, text, size, bold in lines:
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
    if not blocks:
        # Every line looked like a heading (e.g. all-bold slides): keep the text per page
        pages: dict[int, list[str]] = {}
        for page, _, text, _, _ in lines:
            pages.setdefault(page, []).append(text)
        blocks = [Block(p, None, "\n".join(t)) for p, t in pages.items()]
    return blocks


def text_blocks(text: str) -> list[Block]:
    """Non-PDF sources become one block without page or heading."""
    return [Block(None, None, text)] if text and text.strip() else []
