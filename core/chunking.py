"""
StudyMind — Chunkers: fixed token windows and structure-aware (by heading, then size).

Tokens are counted with the bge tokenizer so a 512-token chunk fits bge's window.
"""

from functools import lru_cache

from core.config import settings
from core.parse import Block

CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    tok.no_truncation()
    return tok


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets of each token."""
    return _tokenizer().encode(text, add_special_tokens=False).offsets


def fixed(blocks: list[Block], size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[Block]:
    """Sliding windows over all blocks; each chunk keeps the page/heading where it starts."""
    text, starts = "", []
    for b in blocks:
        if text:
            text += "\n\n"
        starts.append((len(text), b))
        text += b.text
    spans = token_spans(text)

    chunks = []
    step = size - overlap
    for i in range(0, len(spans), step):
        window = spans[i:i + size]
        start = window[0][0]
        origin = next(b for pos, b in reversed(starts) if pos <= start)
        chunks.append(Block(origin.page, origin.section_heading, text[start:window[-1][1]]))
        if i + size >= len(spans):
            break
    return chunks


def structure(blocks: list[Block], max_tokens: int = CHUNK_TOKENS) -> list[Block]:
    """Never cross a heading change; pack whole blocks up to max_tokens."""
    chunks: list[Block] = []
    current: Block | None = None
    current_tokens = 0
    for b in blocks:
        n = len(token_spans(b.text))
        if n == 0:
            continue
        fits = current and current.section_heading == b.section_heading and current_tokens + n <= max_tokens
        if fits:
            current.text += "\n\n" + b.text
            current_tokens += n
            continue
        if current:
            chunks.append(current)
            current = None
        if n > max_tokens:
            chunks.extend(fixed([b], max_tokens, CHUNK_OVERLAP))
        else:
            current, current_tokens = Block(b.page, b.section_heading, b.text), n
    if current:
        chunks.append(current)
    return chunks


def chunk(blocks: list[Block]) -> list[Block]:
    return fixed(blocks) if settings.chunker == "fixed" else structure(blocks)
