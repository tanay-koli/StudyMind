# Phase 4 — Multimodal material: design

Date: 2026-10-29. Builds on Phase 3 (branch `phase-3`, not yet merged into main).

## Goal

Make diagrams, tables and lecture videos searchable and citable. Done when a question that only a
diagram can answer returns the right figure and page.

## Decisions

| Question | Decision |
|---|---|
| Vision model | `ollama/qwen2.5vl:7b` (about 6 GB; one-time `ollama pull`), setting `vision_model`. |
| Which figures | Embedded images (`page.get_image_info`) and vector diagrams (`page.cluster_drawings`), each rendered to a PNG crop. |
| When to caption | In a background worker, in the same loop as concept extraction, so there is one LLM queue. |
| Tables | PyMuPDF `page.find_tables()` → a Markdown chunk. No Docling. |
| YouTube | Chunks keep their first snippet's start time and are cited as `@ 12:34` with a `?t=` deep link. |
| ColPali | Out of scope (a stretch goal in the plan; compared in Phase 6 if at all). |
| Old PDFs | Not backfilled. Re-adding a source gives it figures and tables (stated in the README). |

## Constraints

- Everything local; tests offline with fakes. Synthetic PDFs are built with PyMuPDF inside the tests.
- No migrations: new tables only (`create_all()`); `chunk` keeps its columns.
- Figure extraction must never fail an ingest: a bad crop is skipped and the text still ingests.
- Write files as UTF-8 explicitly.

## Storage (core/db.py)

```python
class ChunkMedia(SQLModel, table=True):
    chunk_id: int = Field(primary_key=True)
    kind: str                         # figure | table | video
    image_path: Optional[str] = None  # figure: crop PNG
    bbox: Optional[str] = None        # figure: "x0,y0,x1,y1" in PDF points
    start_s: Optional[float] = None   # video
    url: Optional[str] = None         # video: https://youtu.be/<id>?t=<int seconds>

class PendingFigure(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(index=True)
    topic_id: str
    page: int
    section_heading: Optional[str] = None
    bbox: str
    image_path: str
    attempts: int = 0
```

Crops live at `data/figures/<source_id>/p<page>_<n>.png` (`FIGURE_DIR = DATA_DIR / "figures"`).
`topics._purge` also deletes the sources' `ChunkMedia` rows (by chunk id), their `PendingFigure` rows,
and the `data/figures/<source_id>/` folders. `move_source` also updates `PendingFigure.topic_id`.

## Parsing (core/parse.py)

`Block` gains two fields with defaults, so existing callers are unchanged:
`kind: str = "text"` (`text` | `table`) and `start_s: Optional[float] = None`.

**Tables.** For each page, `page.find_tables()`. Each table becomes `Block(page, heading, table.to_markdown(), kind="table")`.
A text line whose box lies inside a table's box is not added to prose blocks.

**Figures.** A new function, `extract_figures(path) -> list[FigureCrop]`, with
`FigureCrop(page: int, bbox: tuple[float, float, float, float], png: bytes)`:
- Candidates are the boxes of embedded images (`get_image_info()`) plus `cluster_drawings()` rectangles.
- Candidates are skipped when they are:
  - under `MIN_FIGURE_SIDE = 0.08` of the page width or height, or under `MIN_FIGURE_AREA = 0.03` of the page area
    (a 15% side minimum would drop ordinary wide-and-short flowcharts);
  - over `MAX_FIGURE_AREA = 0.9` of the page area (a background);
  - overlapping a detected table.
- Overlapping candidates merge into their union. Each page keeps at most `MAX_FIGURES_PER_PAGE = 4`, the largest first.
- Each crop is `page.get_pixmap(clip=bbox, dpi=150).tobytes("png")`.
- The section heading is assigned at ingest from the parsed blocks: the heading of the first block on the
  figure's page, else that of the last block before it. Heading detection stays in `parse_pdf` only.
- The whole function is wrapped: any exception on a page is logged and that page is skipped.

**Chunking (core/chunking.py).** `structure()` merges two blocks only when both are `kind == "text"`.
Each chunk keeps its first block's `start_s`. `fixed()` keeps the origin block's `kind` and `start_s`.

## Ingest (core/ingest.py, core/retrieval.py)

- `retrieval.index_chunks(source_id, topic_id, blocks, video_url=None, figure=None)` writes a `ChunkMedia` row
  - for each block with `kind == "table"` (kind=table);
  - for each block with `start_s is not None` (kind=video, `url = f"{video_url}?t={int(start_s)}"`);
  - for every block when `figure={"image_path", "bbox"}` is given (kind=figure). The caption worker passes one block.
- `ingest_pdf` calls `extract_figures`, saves the PNGs, adds `PendingFigure` rows, sets
  `result["figures_queued"]`, and wakes the worker.
- `ingest_youtube` builds one `Block(None, None, snippet.text, start_s=snippet.start)` per snippet and
  passes `video_url=f"https://youtu.be/{video_id}"`.

## Captioning (core/figures.py)

`caption_pending() -> int` handles each `PendingFigure` in id order:

- It calls `llm.complete(prompt, system=..., images=[png], model=settings.vision_model, timeout=180)`. The prompt is:
  "Describe this diagram for a student; list its labels and the relationships between them." plus
  "It appears under the heading: <heading>" when there is one.
- **Success** (a caption of 20 characters or more after stripping): `retrieval.index_chunks(source_id, topic_id,
  [Block(page, heading, f"Figure (p.{page}): {caption}")], figure={"image_path": ..., "bbox": ...})`.
  The pending row is then deleted.
- **Empty or short caption:** `attempts += 1`. At `MAX_ATTEMPTS = 3` the row is deleted and a warning is logged.
- **Exception from the LLM call** (Ollama down, model not pulled): the pass stops without counting an attempt
  and logs one warning. The next wake retries.

`llm.complete` gains `images: list[bytes] | None = None`. When it is given, the user message content
becomes `[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`.

Worker: `concepts._run` awaits `figures.caption_pending()` before `extract_pending()` on each pass,
so new figure chunks also get concept triples. `figures.pending_count(source_ids=None) -> int` exists for messages.

## Retrieval and citations

- `retrieval._hydrate` left-joins `ChunkMedia`. `Hit` gains `media: Optional[dict] = None`
  (`{"kind", "image_path", "start_s", "url"}`), a last field with a default, so positional constructors still work.
- `query._context` adds the label to the passage header: ` — Figure` / ` — Table` / ` @ M:SS` for video.
  Each citation dict gains `kind` (`text` when there is no media), `image_url` (`/figures/<chunk_id>`
  for figures, else None) and `url` (the video deep link, else None).
- The agent stores hits with `dataclasses.asdict`, so `media` is carried, and `Hit(**d)` rebuilds them.

## API

- `GET /figures/{chunk_id}` returns a `FileResponse(image_path, media_type="image/png")` for a figure
  `ChunkMedia` row whose file exists; otherwise 404. The path comes only from the database.
- `/health` adds `vision` with the same check as `llm` (`check_llm` takes the model name).
- The ingest PDF response carries `figures_queued`.

## UI

- `_render_citations`: a figure chip is `[n] 🖼️ <title> · p.N`; its popover shows `st.image(api_url + image_url)`
  and the caption. A video chip is `[n] 🎬 <title> @ M:SS`; its popover shows the text and a link to `url`.
- Below an answer, in the Ask and Agent tabs, `_render_figures(answer, citations)` shows the images of up to
  2 figure citations whose `[n]` appears in the answer text.
- The ingest success message adds "N figures queued for captioning — searchable in a few minutes."

## Testing (offline)

1. `extract_figures` on a synthetic page with a vector diagram, an embedded PNG and a ruled table finds the
   2 figures and not the table. A full-page background rectangle and a small logo are skipped.
2. `parse_pdf` gives one `kind="table"` Markdown block; the table's cell text is not in any prose block.
3. `caption_pending`:
   - success creates the Chunk, a `ChunkMedia(kind=figure)` and the Qdrant point, and removes the pending row;
   - an empty caption counts an attempt, and the row is dropped at 3;
   - an LLM exception leaves `attempts` at 0 and the row in place.
4. Deleting a source removes its crop folder, pending rows and `ChunkMedia` rows.
5. YouTube ingest with fake snippets: each chunk's `ChunkMedia.start_s` is its first snippet's start, and its
   `url` ends with `?t=<int>`.
6. Citations carry `kind`, `image_url` and `url`; `_context` labels figures, tables and video times.
   `GET /figures/{id}` returns PNG bytes, and 404 for unknown ids or non-figure chunks.
7. `llm.complete(images=[...])` sends a content list with a base64 `image_url` part (fake `litellm.acompletion`).
8. Done-when, offline: page 1 is unrelated prose and page 2 has only a diagram. With a fake caption naming
   its labels, `ask` and the agent both return a top citation with `page == 2`, `kind == "figure"` and an `image_url`.

Done-when, real model, checked by hand: pull `qwen2.5vl:7b`, ingest a real lecture PDF with a diagram on
throwaway `DATABASE_URL` / `QDRANT_PATH` / `AGENT_DB_PATH` paths, and ask a question only the diagram answers.

## Out of scope

- ColPali and CLIP retrieval.
- Backfilling figures for PDFs already ingested.
- OCR of scanned pages.
- Captions for images in web articles.
- Figure deduplication across sources.
