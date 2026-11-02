# Phase 4 — Multimodal Material Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PDF diagrams (raster and vector) are captioned by a local vision model and become citable figure chunks with an inline image. Tables become Markdown chunks. YouTube chunks cite `@ M:SS` with a deep link.

**Architecture:** `parse.py` finds tables and figure crops. Ingest stores the crops as `PendingFigure` rows. The existing background worker captions them with `ollama/qwen2.5vl:7b` and indexes each caption as a chunk, with a `ChunkMedia` side row (figure, table or video). Retrieval left-joins `ChunkMedia`, so every citation carries `kind` / `image_url` / `url`. The API serves crops at `/figures/{chunk_id}` and the UI renders them.

**Tech Stack:** PyMuPDF 2.0 (`find_tables`, `cluster_drawings`, `get_image_info`, `get_pixmap`), LiteLLM → Ollama (`qwen2.5vl:7b`), SQLModel, FastAPI, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-phase4-multimodal-design.md`

## Global Constraints

- Everything local; tests offline with the `tests/conftest.py` fakes. No network and no model downloads in tests.
- New tables only (`ChunkMedia`, `PendingFigure`); never add columns to existing tables.
- Figure extraction never fails an ingest: catch per page, log, skip.
- `vision_model: str = "ollama/qwen2.5vl:7b"`; `FIGURE_DIR = DATA_DIR / "figures"`; crop path `FIGURE_DIR / str(source_id) / f"p{page}_{n}.png"`; 150 dpi.
- Figure filters (as amended in the spec; the original 15% width/height minimum dropped wide-and-short flowcharts): skip when either side is under `MIN_FIGURE_SIDE = 0.08` of the page side, or the area is under `MIN_FIGURE_AREA = 0.03` of the page; skip over `MAX_FIGURE_AREA = 0.9`; skip when overlapping a table; `MAX_FIGURES_PER_PAGE = 4`, largest first.
- Captions: success needs at least `MIN_CAPTION = 20` characters after stripping. `MAX_ATTEMPTS = 3`. Chunk text is `f"Figure (p.{page}): {caption}"`.
- Async tests use `@pytest.mark.asyncio`. Fake `llm.complete` replacements accept `(prompt, system=None, **kw)`, and `images` and `model` arrive in `kw`.
- Write files as UTF-8 and edit with the Edit tool, not shell heredocs. Heredocs mangled `\n` and `\b` twice in Phase 3.
- Baseline is 167 passing tests. Every task ends with the full suite green: `.venv/Scripts/python -m pytest -q`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Scanned PDF** (pages are one full-page image, with no text): the full-page image is skipped (over 90% of the area), the text error "PDF contains no extractable text." stays, and no figures are queued. Test in Task 4.
2. **Deleting a source while its figures are still pending:** the pending rows and the crop folder go, and the worker never indexes a figure for a deleted source. Test in Task 5 (caption a row whose source was deleted → row removed, no chunk).
3. **Ollama running but `qwen2.5vl` not pulled:** the pass stops with no attempts counted, and `/health` shows `vision.ok == false` with the pull command. Tests in Tasks 1 and 5.
4. **A YouTube video with no `start` on its snippets** (old API dicts): the chunks get no video `ChunkMedia` and ingest still succeeds. Test in Task 6.
5. **A figure chunk whose PNG was deleted from disk:** `GET /figures/{id}` returns 404, not 500, and the UI chip still renders its caption. Test in Task 7.

---

### Task 1: Images through `llm.complete`; vision setting and health

**Files:**
- Modify: `core/llm.py` (`LLMClient.complete`), `core/config.py` (`Settings`, `check_llm`), `api/main.py` (`/health`), `ui/app.py:257-262` (status list)
- Test: `tests/test_llm.py`, `tests/test_api.py::test_health`

**Interfaces:**
- Produces: `llm.complete(prompt, system=None, images: list[bytes] | None = None, **kw)`; `settings.vision_model`; `check_llm(model: str | None = None, timeout: int = 5) -> dict` (`model` defaults to `settings.llm_model`); `/health` returns `{"llm": ..., "vision": ...}`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_llm.py
@pytest.mark.asyncio
async def test_complete_sends_images_as_data_urls(monkeypatch):
    seen = {}
    async def fake_acompletion(model, messages, **kw):
        seen.update(model=model, messages=messages)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    await LLMClient().complete("describe", images=[b"\x89PNG"], model="ollama/qwen2.5vl:7b")
    content = seen["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
    assert seen["model"] == "ollama/qwen2.5vl:7b"
```
(Read `tests/test_llm.py` first and reuse its existing fake-acompletion style if it has one. The real `llm.complete` is replaced by the autouse fixture, so construct a fresh `LLMClient()`.)

```python
# tests/test_api.py — replace test_health
def test_health(client):
    body = client.get("/health").json()
    assert list(body) == ["llm", "vision"] and body["llm"]["ok"]
    assert body["vision"]["ok"] is False and "ollama pull qwen2.5vl:7b" in body["vision"]["error"]
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest -q tests/test_llm.py tests/test_api.py -k "images or health"`. Expected: both FAIL.
- [ ] **Step 3: Implement.** When `images` is given, the user content is the list from the spec (base64 PNG data URLs). `/health` runs `check_llm()` and `check_llm(settings.vision_model)`. The UI status list adds `(f"Vision ({settings.vision_model})", st.session_state.health["vision"])`.
- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Optional real check.** Only if `ollama list` already shows `qwen2.5vl:7b`: run a 5-line script in the scratchpad that calls `llm.complete("What colour is this?", images=[<64x64 red PNG from pymupdf.Pixmap>], model=settings.vision_model)` and prints the reply. Expected: it mentions red. If the model isn't pulled, skip and note it in the ledger; Task 8 covers it.
- [ ] **Step 6: Commit** `feat: llm.complete sends images; vision model setting and health check`

---

### Task 2: Storage, Block fields, chunker and index_chunks media

**Files:**
- Modify: `core/db.py` (two tables), `core/parse.py` (`Block`), `core/chunking.py` (`structure`, `fixed`), `core/retrieval.py` (`index_chunks`), `core/topics.py` (`_purge`, `move_source`), `core/config.py` (`FIGURE_DIR`)
- Test: `tests/test_chunking.py`, `tests/test_retrieval.py`, `tests/test_topics.py`

**Interfaces:**
- Produces:
  - `ChunkMedia` and `PendingFigure`, exactly as in the spec.
  - `Block(page, section_heading, text, kind="text", start_s=None)`.
  - `index_chunks(source_id, topic_id, blocks, video_url: str | None = None, figure: dict | None = None) -> int`.
  - `config.FIGURE_DIR`.

- [ ] **Step 1: Failing tests**
  - `test_structure_never_merges_table_into_prose`: three blocks on the same page and heading (text, table, text) → 3 chunks, the middle one `kind == "table"`.
  - `test_chunks_keep_first_start_s`: blocks with `start_s` 0.0, 4.5, 9.0 that fit in one chunk → 1 chunk with `start_s == 0.0`. A `fixed()` window keeps its origin block's `start_s`.
  - `test_index_chunks_writes_media`:
    - a table block gives `ChunkMedia(kind="table")`;
    - a `start_s=75.4` block with `video_url="https://youtu.be/abc"` gives `kind="video"`, `start_s=75.4`, `url="https://youtu.be/abc?t=75"`;
    - `figure={"image_path": "x.png", "bbox": "1,2,3,4"}` gives `kind="figure"` with those values;
    - a plain text block gives no row.
  - `test_delete_source_removes_media_pending_and_crops`:
    - set up a source with a figure-media chunk, a `PendingFigure` row and a file at `FIGURE_DIR/<sid>/p1_0.png` (monkeypatch `config.FIGURE_DIR` / the module constant to `tmp_path`);
    - `delete_source` → no `ChunkMedia`, no `PendingFigure`, and the folder is gone.
  - `test_move_source_moves_pending_figures`: `PendingFigure.topic_id` follows `move_source`.
- [ ] **Step 2: Run them.** Expected: FAIL (`kind` is an unexpected keyword / ImportError).
- [ ] **Step 3: Implement.** The `ChunkMedia` rows are added in the same session as the `Chunk` rows, after the refresh gives ids, before `_upsert`. `_purge` uses `shutil.rmtree(FIGURE_DIR / str(sid), ignore_errors=True)` per source. That runs after the row deletes, in the same function; a leftover folder only costs disk.
- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: ChunkMedia/PendingFigure tables; blocks carry kind and start time`

---

### Task 3: Tables as Markdown chunks

**Files:**
- Create: `tests/pdfs.py` (synthetic PDF builders, shared by Tasks 3–7)
- Modify: `core/parse.py` (`parse_pdf`)
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces: `tests.pdfs.figure_pdf(path) -> Path`. Page 1 has a heading, prose, a ruled 3×3 table (cells `Enzyme`, `Rate`, …, numbers), a vector diagram (two boxes `citrate` / `isocitrate` joined by a line, total box about 380×160 pt), an embedded 64×64 PNG drawn at 150×150 pt, and a 30×30 pt logo rectangle. Page 2 is a full-page filled rectangle with one short line of text. `tests.pdfs.diagram_only_pdf(path) -> Path`: page 1 is unrelated prose about photosynthesis; page 2 has only a heading and the citrate/isocitrate diagram.

- [ ] **Step 1: Failing test** `test_table_becomes_one_markdown_block`: `parse_pdf(figure_pdf(...))` has exactly one `kind == "table"` block. Its text contains `|` and `Enzyme`, and no `kind == "text"` block contains `Enzyme`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `find_tables()` per page. A line is dropped from prose when its bbox centre lies inside a table bbox; `_lines` must also yield each line's bbox. Table blocks get the current heading.
- [ ] **Step 4: Full suite.** Expected: all pass. The existing parse tests stay green, because pages without tables are unchanged.
- [ ] **Step 5: Commit** `feat: PDF tables become Markdown chunks`

---

### Task 4: `extract_figures`

**Files:**
- Modify: `core/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces: `@dataclass FigureCrop(page: int, bbox: tuple[float, float, float, float], png: bytes)` and `extract_figures(path: str) -> list[FigureCrop]`.
- `section_heading` is not on `FigureCrop` (spec): Task 5's ingest assigns it from the parsed blocks (the heading of the first block on that page, else the last block before it), so heading detection isn't duplicated.

- [ ] **Step 1: Failing tests**
  - `test_extract_figures_finds_diagram_and_image_not_table`: on `figure_pdf`, exactly 2 crops on page 1. One bbox contains the diagram boxes and one contains the image rect. None overlaps the table, there's no crop on page 2 (the background), and the 30 pt logo is skipped. Each `png` starts with `b"\x89PNG"`.
  - `test_scanned_page_queues_no_figures` (Review Focus 1): a page that is one full-page image gives `[]`.
  - `test_extract_figures_survives_a_bad_page`: monkeypatch `pymupdf.Page.cluster_drawings` to raise on the first call → the other pages' crops are still returned.
- [ ] **Step 2: Run.** Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** with the filters in Global Constraints. Merge overlapping candidates into their union by repeating until nothing changes, then sort by area descending and keep at most 4. Crop with `page.get_pixmap(clip=bbox, dpi=150).tobytes("png")`.
- [ ] **Step 4: Full suite.** Expected: all pass. If the synthetic geometry and PyMuPDF's real clustering disagree (for example, the diagram splits into two clusters), adjust the test PDF geometry, not the filters, and record a ruling.
- [ ] **Step 5: Commit** `feat: extract figure crops (embedded images and vector diagrams)`

---

### Task 5: Queue figures at ingest; caption worker

**Files:**
- Create: `core/figures.py`
- Modify: `core/ingest.py` (`ingest_pdf`), `core/concepts.py` (`_run`)
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `extract_figures`, `FigureCrop` (Task 4); `index_chunks(..., figure=)` and `PendingFigure` (Task 2); `llm.complete(images=, model=)` (Task 1).
- Produces: `figures.caption_pending() -> int` (figures indexed); `figures.pending_count(source_ids: list[int] | None = None) -> int`; `ingest_pdf` result key `figures_queued: int`; constants `MIN_CAPTION`, `MAX_ATTEMPTS`, `CAPTION_PROMPT`.

- [ ] **Step 1: Failing tests** (`tests/test_figures.py`; monkeypatch `figures.FIGURE_DIR` or the config constant to `tmp_path`)
  - `test_ingest_pdf_queues_figures`: `ingest_pdf(figure_pdf)` → `figures_queued == 2`. There are 2 `PendingFigure` rows, each with a heading and a PNG file on disk.
  - `test_caption_success_indexes_figure_chunk`:
    - a fake `llm.complete` returns `"A diagram where citrate converts to isocitrate in the Krebs cycle."`;
    - `caption_pending() == 2`, there are no pending rows, and 2 chunks have text starting `"Figure (p.1): "`;
    - each has a `ChunkMedia(kind="figure")` with an existing `image_path`;
    - the fake got `images=[png]` and `model == settings.vision_model`.
  - `test_short_caption_counts_attempts_and_drops_at_three`: reply `"ok"` → after three passes the row is gone, no chunk exists, and `attempts` read 1 and 2 in between.
  - `test_llm_error_leaves_row_untouched` (Review Focus 3): the fake raises `ConnectionError` → `caption_pending() == 0`, the row is still there with `attempts == 0`, and the second row isn't tried.
  - `test_pending_for_deleted_source_is_dropped` (Review Focus 2): delete the source after ingest, then `caption_pending()` → no chunk and no pending rows. It's covered by the purge from Task 2, and this test proves the worker doesn't resurrect it.
  - `test_worker_captions_before_concepts`: monkeypatch `figures.caption_pending` and `concepts.extract_pending` to append to a list. One `_run` pass (run `start()`, `wake()`, then `stop()`) records `["figures", "concepts"]` first.
- [ ] **Step 2: Run.** Expected: FAIL (no `core.figures`).
- [ ] **Step 3: Implement.** `ingest_pdf`:
  - runs `extract_figures` in `asyncio.to_thread` after `_store` succeeds;
  - writes the PNGs and `PendingFigure` rows;
  - wraps all of it in `try/except Exception` that logs and sets `figures_queued = 0`;
  - then calls `concepts.wake()`.

  `caption_pending` re-reads each row before indexing; if it's gone (source deleted), it skips. `concepts._run` awaits `figures.caption_pending()` inside the existing try, before `extract_pending()`.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: queue figure crops at ingest; worker captions them with the vision model`

---

### Task 6: YouTube timestamps

**Files:**
- Modify: `core/ingest.py` (`ingest_youtube`)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Block(start_s=)` and `index_chunks(video_url=)` (Task 2).

- [ ] **Step 1: Failing tests** (monkeypatch `youtube_transcript_api.YouTubeTranscriptApi` with a fake whose `fetch` returns snippet objects `SimpleNamespace(text=..., start=..., duration=...)`)
  - `test_youtube_chunks_cite_start_time`: 3 snippets at 0.0 / 12.5 / 75.2 that fit in one chunk → one chunk with `ChunkMedia(kind="video", start_s=0.0, url="https://youtu.be/<id>?t=0")`. With the chunk size monkeypatched small (`chunking.CHUNK_TOKENS` → 5), the second chunk's `start_s` is its own first snippet's start.
  - `test_youtube_without_start_times_still_ingests` (Review Focus 4): snippets as dicts with only `text` → success, and no video media rows.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** One block per snippet with `start_s=getattr(snippet, "start", None)`; pass `video_url=f"https://youtu.be/{video_id}"` through `_store`. `_store` gains `**index_kw`, forwarded to `index_chunks`.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: YouTube chunks keep their start time and deep link`

---

### Task 7: Media in hits and citations; figure endpoint; done-when

**Files:**
- Modify: `core/retrieval.py` (`Hit`, `_hydrate`, `_hit`), `core/query.py` (`_context`), `api/main.py` (`GET /figures/{chunk_id}`)
- Test: `tests/test_query.py`, `tests/test_api.py`, `tests/test_figures.py`

**Interfaces:**
- Produces:
  - `Hit.media: Optional[dict] = None`, the last field, with keys `kind, image_path, start_s, url`.
  - Citation dicts gain `kind` (`"text"` default), `image_url` (`f"/figures/{chunk_id}"` for figures, else `None`) and `url`.
  - Passage header labels: ` — Figure`, ` — Table`, ` @ M:SS` (`H:MM:SS` from an hour).

- [ ] **Step 1: Failing tests**
  - `test_context_labels_media_and_citations_carry_it`: hits with figure, table and video media (video `start_s=754`) → the headers contain `— Figure`, `— Table` and `@ 12:34`. Citations have `kind` and `image_url == "/figures/<id>"` for the figure and `url` for the video, and `kind == "text"` / `None` / `None` for a plain hit.
  - `test_figure_endpoint` (Review Focus 5): it returns `image/png` bytes for a figure chunk; 404 for an unknown id, for a table chunk, and for a figure whose file was deleted.
  - `test_diagram_question_returns_figure_and_page` (the offline done-when):
    - `ingest_pdf(diagram_only_pdf)`, then `caption_pending()` with the fake caption `"Flow diagram: citrate is converted to isocitrate by aconitase."`;
    - `query.ask("what converts citrate to isocitrate?")` → `citations[0]` has `page == 2`, `kind == "figure"` and an `image_url`;
    - running the agent graph (`agent.open_graph(tmp_path / "a.db")`, fake `llm.complete` replies) gives the same citation first.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `_hydrate` selects `ChunkMedia` with `isouter=True`. The endpoint looks up `ChunkMedia` by id and serves only `kind == "figure"` rows whose `Path(image_path).is_file()`.
- [ ] **Step 4: Full suite.** Expected: all pass. The eval runner, Phase 3 agent tests and `Hit(...)` positional constructors keep working.
- [ ] **Step 5: Commit** `feat: citations carry figures, tables and video times; /figures endpoint`

---

### Task 8: UI, README, real-model check

**Files:**
- Modify: `ui/app.py` (`_render_citations`, new `_render_figures`, Ask and Agent answer rendering, PDF ingest success message), `README.md` (features table, `.env` block `VISION_MODEL`, Phase 4 usage notes, `ollama pull qwen2.5vl:7b`, "re-add older PDFs to get figures"), `.gitignore` (`data/figures/`)

**Interfaces:**
- Consumes: citation `kind` / `image_url` / `url` (Task 7); `figures_queued` (Task 5).

- [ ] **Step 1: Implement.**
  - Chips: `[n] 🖼️ title · p.N` for figures, `[n] 🎬 title @ M:SS` for video, unchanged for text and tables.
  - Figure popover: `st.image(settings.api_url + image_url)` plus the text. Video popover: `st.link_button("Open at M:SS", url)` plus the text.
  - `_render_figures(answer, citations)`: up to 2 figure citations whose `[n]` occurs in `answer`, as `st.image(..., caption=f"[{n}] {title} · p.{page}")`, called after the answer in both tabs.
  - PDF ingest success shows `figures_queued`.
  - A failed image load must not break the page; `st.image` with a URL leaves loading to the browser.
- [ ] **Step 2: Offline check.** Run `.venv/Scripts/python -m py_compile ui/app.py`, then the full suite. Expected: compiles and all pass.
- [ ] **Step 3: Real-model done-when** (by hand; ask the user before the roughly 6 GB `ollama pull qwen2.5vl:7b` if it isn't pulled yet).
  - Use throwaway `DATABASE_URL` / `QDRANT_PATH` / `AGENT_DB_PATH`, the API on a spare port, and stop it with the background-task stop tool, never `taskkill` by image name.
  - Ingest a real lecture PDF with a diagram (or `diagram_only_pdf` if none is at hand) and wait until `pending_count() == 0`.
  - Ask a question only the diagram answers. Pass: the first citation is the figure on the right page, and its `/figures/{id}` returns the crop.
  - Record the caption, timings and pass/fail in the ledger.
- [ ] **Step 4: Commit** `feat: figures inline and video timestamps in the UI; README phase 4`
