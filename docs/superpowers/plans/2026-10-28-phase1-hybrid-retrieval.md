# Phase 1 — Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Own ingestion (structured parse → chunk → embed → Qdrant) and a `HybridRetriever` (dense + BM25 → RRF → cross-encoder rerank) behind a `Retriever` interface, with page-cited scoped answers and real deletes.

**Architecture:** Three new focused modules (`core/parse.py`, `core/chunking.py`, `core/retrieval.py`). Ingest writes `Chunk` rows + Qdrant points first and uploads to Cognee best-effort. `ask`/`ask_scoped` retrieve through `get_retriever()` and generate with `llm.complete`. Delete/move in `core/topics.py` keep Qdrant and SQLite in sync.

**Tech Stack:** Python 3.13, PyMuPDF, `qdrant-client[fastembed]` (embedded mode), fastembed ONNX models, `tokenizers`, SQLModel, LiteLLM, FastAPI, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-phase1-hybrid-retrieval-design.md`

## Global Constraints

- Tests run offline: no model downloads, no network. Conftest fakes tokenizer, embedders, reranker; Qdrant is `:memory:`.
- Run tests with `.venv/Scripts/python -m pytest -q`. All 28 existing tests keep passing (updated only where this plan says).
- Dense models: `bge-small` → `BAAI/bge-small-en-v1.5` (384-d, no prefixes); `nomic` → `nomic-ai/nomic-embed-text-v1.5` (768-d, prefixes `search_query: ` / `search_document: `).
- Sparse: fastembed `Qdrant/bm25`; documents via `.embed`, queries via `.query_embed`; collection sparse config uses `modifier=models.Modifier.IDF`.
- Reranker: fastembed `TextCrossEncoder("BAAI/bge-reranker-base")`.
- Fusion: RRF with k = 60 over dense top 30 and sparse top 30; rerank fused top 30 → 6.
- Fixed chunker: 512 tokens, 64 overlap. Structure chunker: max 512 tokens. Tokens counted with the bge tokenizer (`add_special_tokens=False`, truncation disabled).
- Collections: `chunks_bge_small`, `chunks_nomic`. Point id = `Chunk.id`; named dense vector `dense`; sparse vector `bm25`; payload `{source_id, topic_id}` with payload indexes on both.
- Settings (`.env`): `EMBED_MODEL=bge-small`, `CHUNKER=structure`, `RETRIEVER=hybrid`, `QDRANT_PATH=data/qdrant`.
- New dependency pinned with `==` in `requirements.txt` under a `# Retrieval` heading.
- Small diffs, match existing style (module docstrings, section comment bars, `dict` results with `status`/`error`).

## Review Focus

1. Asking a topic that has no chunks yet (or before any collection exists) → friendly "No results" answer, empty `citations`, no exception. Test in Task 4 and Task 6.
2. Indexing fails mid-ingest (embedder raises) → status `error`, the `Source` row is rolled back, no `Chunk` rows or points left behind. Test in Task 5.
3. Switching `EMBED_MODEL` after ingesting → delete still removes points from every `chunks_*` collection; `reindex()` fills the new collection from `Chunk` rows. Test in Task 3.
4. LLM unavailable during ask → `error` set, no crash, HTTP 200 like other errors. Test in Task 6.
5. Tiny inputs and oversize blocks: text shorter than the overlap, a single block over 512 tokens in the structure chunker, whitespace-only blocks. Test in Task 2.

---

### Task 1: Structured PDF parser

**Files:**
- Create: `core/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces: `@dataclass Block(page: int | None, section_heading: str | None, text: str)`; `parse_pdf(path: str) -> list[Block]`; `text_blocks(text: str) -> list[Block]` (one block, `page=None`, `section_heading=None`; empty list for blank text).

- [ ] **Step 1: Write failing tests** — build PDFs in-test with PyMuPDF (`page.insert_text(..., fontsize=...)`; bold via `fontname="hebo"`).

```python
def test_headings_and_pages(tmp_path):
    # page 1: "Attention" at fontsize 20, body at 11; page 2: "Transformers" at 20, body at 11
    blocks = parse_pdf(str(pdf))
    assert {(b.page, b.section_heading) for b in blocks} == {(1, "Attention"), (2, "Transformers")}
    assert all(b.text.strip() and "Attention" not in b.text.split("\n")[0] for b in blocks)

def test_bold_short_line_is_heading(tmp_path):
    # "Key idea" in hebo 11pt, then body 11pt
    assert parse_pdf(str(pdf))[0].section_heading == "Key idea"

def test_heading_carries_to_next_page(tmp_path):
    # heading on page 1, body only on page 2
    last = parse_pdf(str(pdf))[-1]
    assert (last.page, last.section_heading) == (2, "Attention")

def test_text_blocks():
    assert text_blocks("  ") == []
    assert text_blocks("hello") == [Block(None, None, "hello")]
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest -q tests/test_parse.py` — expected FAIL (module missing).

- [ ] **Step 3: Implement `core/parse.py`.** Use `page.get_text("dict")`; a line's text is its spans joined, its size the max span size, bold when `span["flags"] & 16`. Body size = median span size weighted by character count over the whole document. A line is a heading when `size >= 1.2 * body` or (bold and ≤ 12 words). Headings set the current heading (carried across pages) and are not included in block text; each PyMuPDF text block's remaining lines become one `Block` (skip blank). `import pymupdf` (not `fitz`, which warns).

- [ ] **Step 4: Run** the tests — expected PASS.

- [ ] **Step 5: Commit** `git add core/parse.py tests/test_parse.py && git commit -m "feat: structured PDF parser with pages and section headings"`

---

### Task 2: Chunkers + dependency + token spans

**Files:**
- Create: `core/chunking.py`
- Modify: `requirements.txt`, `tests/conftest.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `Block` from Task 1.
- Produces: `token_spans(text: str) -> list[tuple[int, int]]` (char offsets per token; module-level so tests patch it); `fixed(blocks: list[Block], size: int = 512, overlap: int = 64) -> list[Block]`; `structure(blocks: list[Block], max_tokens: int = 512) -> list[Block]`; `chunk(blocks: list[Block]) -> list[Block]` (dispatches on `settings.chunker`).

- [ ] **Step 1: Install and pin.** `.venv/Scripts/python -m pip install "qdrant-client[fastembed]"`, then add `qdrant-client[fastembed]==<installed version>` to `requirements.txt` under `# Retrieval`.

- [ ] **Step 2: Conftest fake.** In the `offline` fixture: `monkeypatch.setattr(chunking, "token_spans", lambda t: [m.span() for m in re.finditer(r"\S+", t)])`.

- [ ] **Step 3: Write failing tests** (whitespace tokens via the fake).

```python
def words(n, start=0): return " ".join(f"w{i}" for i in range(start, start + n))

def test_fixed_window_and_overlap():
    out = fixed([Block(1, "A", words(1000))], size=512, overlap=64)
    assert [len(c.text.split()) for c in out] == [512, 512, 104]
    assert out[1].text.split()[0] == "w448"

def test_fixed_takes_page_of_start_block():
    out = fixed([Block(1, "A", words(500)), Block(2, "B", words(500, 500))])
    assert [(c.page, c.section_heading) for c in out] == [(1, "A"), (1, "A"), (2, "B")]  # 2nd window starts at token 448

def test_structure_splits_on_heading_change():
    out = structure([Block(1, "A", "a1"), Block(1, "A", "a2"), Block(2, "B", "b1")])
    assert [(c.page, c.section_heading, c.text) for c in out] == [(1, "A", "a1\n\na2"), (2, "B", "b1")]

def test_structure_splits_oversize_block():
    out = structure([Block(3, "A", words(1200))])
    assert all(len(c.text.split()) <= 512 for c in out) and all(c.page == 3 for c in out)

def test_tiny_and_blank_inputs():
    assert [c.text for c in fixed([Block(1, None, "hi")])] == ["hi"]
    assert fixed([Block(1, None, "   ")]) == [] and structure([]) == []
```

- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest -q tests/test_chunking.py` — expected FAIL.

- [ ] **Step 5: Implement `core/chunking.py`.** Real `token_spans`: lazily load `tokenizers.Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")`, call `.no_truncation()`, return `encode(text, add_special_tokens=False).offsets`. `fixed`: join block texts with `"\n\n"`, remember each block's start offset, slide windows of `size` token spans with step `size - overlap`, slice the joined text from first span start to last span end; page/heading from the block containing the start offset; stop once a window reaches the last token. `structure`: group consecutive blocks by `section_heading`; inside a group pack whole blocks greedily up to `max_tokens` (joined `"\n\n"`, page of the first packed block); a block alone over `max_tokens` goes through `fixed([block], max_tokens, 64)`.

- [ ] **Step 6: Run** the tests — expected PASS; full suite still green.

- [ ] **Step 7: Commit** `git add core/chunking.py tests/test_chunking.py tests/conftest.py requirements.txt && git commit -m "feat: fixed and structure-aware chunkers"`

---

### Task 3: Chunk table, settings and Qdrant store

**Files:**
- Modify: `core/config.py` (Settings), `core/db.py` (Chunk table), `tests/conftest.py`
- Create: `core/retrieval.py` (store part)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Block` (Task 1).
- Produces:
  - Settings fields `embed_model: str = "bge-small"`, `chunker: str = "structure"`, `retriever: str = "hybrid"`, `qdrant_path: str = (DATA_DIR / "qdrant").as_posix()`.
  - `class Chunk(SQLModel, table=True)`: `id: Optional[int]` pk, `source_id: int` (fk `source.id`, index), `topic_id: str` (index), `chunker: str`, `ord: int`, `page: Optional[int]`, `section_heading: Optional[str]`, `text: str`.
  - In `core/retrieval.py`: `EMBED_MODELS: dict[str, tuple[str, int, str, str]]` (name → model id, dim, query prefix, doc prefix); `embed_dense(texts: list[str], *, query: bool) -> list[list[float]]`; `embed_sparse(texts: list[str], *, query: bool) -> list[models.SparseVector]`; `rerank(query: str, texts: list[str]) -> list[float]` (all three module-level, patched in tests); `store() -> QdrantClient`; `reset_store() -> None`; `collection_name() -> str` (`"chunks_" + embed_model.replace("-", "_")`); `index_chunks(source_id: int, topic_id: str, blocks: list[Block]) -> int` (writes `Chunk` rows, then points; returns count); `delete_source_points(source_ids: list[int]) -> None` (every `chunks_*` collection); `move_source_points(source_id: int, topic_id: str) -> None` (every `chunks_*` collection, `set_payload` by filter); `reindex() -> int` (embeds all `Chunk` rows into the current collection).

- [ ] **Step 1: Conftest fakes.** Set `settings.qdrant_path = ":memory:"`, `settings.embed_model = "bge-small"`; call `retrieval.reset_store()` before and after. Fake `embed_dense`: hashed bag of lowercase words into `EMBED_MODELS[settings.embed_model][1]` dims, plus 1.0 in dim 0, L2-normalised. Fake `embed_sparse`: `SparseVector(indices=sorted unique crc32(word) % 2**20, values=counts)`. Fake `rerank`: number of query words present in the text.

- [ ] **Step 2: Write failing tests.**

```python
def _src(topic="general", title="Doc"):  # adds a Source row via topics.add_source, returns its id

def test_index_writes_rows_and_points():
    sid = _src()
    n = retrieval.index_chunks(sid, "general", [Block(1, "A", "alpha"), Block(2, "B", "beta")])
    assert n == 2
    assert retrieval.store().count(retrieval.collection_name()).count == 2
    # Chunk rows carry page / heading / chunker

def test_sparse_query_respects_topic_filter():
    # spike: verifies embedded Qdrant + IDF sparse + payload filter
    # index "self attention" under topic t1 and "self attention" under t2; sparse query filtered to t1 returns only t1's point

def test_delete_hits_every_collection(monkeypatch):
    sid = _src(); retrieval.index_chunks(sid, "general", [Block(1, None, "alpha")])
    monkeypatch.setattr(settings, "embed_model", "nomic"); retrieval.reindex()
    retrieval.delete_source_points([sid])
    for name in ("chunks_bge_small", "chunks_nomic"):
        assert retrieval.store().count(name).count == 0

def test_move_updates_payload():
    # after move_source_points(sid, "t2"), a filter on topic_id == "t2" counts the point; "general" counts 0
```

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest -q tests/test_retrieval.py` — expected FAIL.

- [ ] **Step 4: Implement.** `store()`: cached `QdrantClient(location=":memory:")` when `qdrant_path == ":memory:"`, else `QdrantClient(path=settings.qdrant_path)`. Create the current collection on first write (`vectors_config={"dense": VectorParams(size=dim, distance=COSINE)}`, `sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}`, keyword payload indexes on `source_id` (integer) and `topic_id`). Real embedders: lazy fastembed singletons; dense texts get the model's query/doc prefix. `index_chunks` stamps `chunker=settings.chunker`, `ord` = position. Skip the "General" special case: topic id is stored as given.

- [ ] **Step 5: Run** the tests — expected PASS. If the sparse+filter spike fails in embedded mode, stop and report before continuing (fallback: `rank_bm25`, needs a spec change).

- [ ] **Step 6: Commit** `git add core/config.py core/db.py core/retrieval.py tests/conftest.py tests/test_retrieval.py && git commit -m "feat: Chunk table and embedded Qdrant store"`

---

### Task 4: Retrievers (RRF, rerank, Cognee baseline)

**Files:**
- Modify: `core/retrieval.py`, `core/query.py` (only to expose nothing new; `CogneeRetriever` imports `_recall`, `_extract_text_from_recall`, `_extract_sources_from_recall` from it lazily to avoid a cycle)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: store functions from Task 3.
- Produces: `@dataclass Hit(chunk_id: int | None, source_id: int | None, source_title: str, page: int | None, section: str | None, text: str, score: float)`; `class Retriever(Protocol): async def retrieve(self, query: str, topic_id: str | None, k: int = 6) -> list[Hit]`; `rrf(rankings: list[list[int]], k: int = 60) -> list[int]` (ids by fused score, desc; ties keep first-seen order); `HybridRetriever`; `CogneeRetriever`; `get_retriever() -> Retriever`.

- [ ] **Step 1: Write failing tests.**

```python
def test_rrf():
    assert rrf([[1, 2, 3], [3, 2, 4]]) == [3, 2, 1, 4]  # 3: 1/63+1/61 > 2: 2/62 > 1: 1/61 > 4: 1/63

async def test_hybrid_returns_relevant_chunk_with_page():
    # index under topic t1: Block(4, "Attention", "self attention weighs tokens"), Block(5, "RNN", "recurrent hidden state")
    hits = await HybridRetriever().retrieve("what is self attention", "t1", k=1)
    assert (hits[0].page, hits[0].section, hits[0].source_title) == (4, "Attention", "Doc")

async def test_hybrid_topic_none_searches_all_topics(): ...  # hits from t1 and t2

async def test_hybrid_empty_store_returns_empty():
    assert await HybridRetriever().retrieve("anything", "t1") == []

async def test_cognee_retriever_uses_chunks_search(offline):
    hits = await CogneeRetriever().retrieve("q", None)
    assert hits[0].text == RECALL_TEXT and hits[0].page is None
    assert "/api/v1/recall" in offline
```

- [ ] **Step 2: Run** — expected FAIL.

- [ ] **Step 3: Implement.** `HybridRetriever.retrieve` runs one `asyncio.to_thread` of a sync search: return `[]` if the collection doesn't exist; dense and sparse `query_points(..., using="dense"|"bm25", query_filter=topic filter or None, limit=30)`; `rrf` → top 30 ids; load `Chunk` rows joined to `Source.title`; `rerank(query, texts)`; sort desc; top `k` → `Hit` with rerank score. `CogneeRetriever`: `_recall(query, search_type="CHUNKS", datasets=get_topic_datasets(topic_id) or None if topic_id else None)`, one `Hit` per extracted text (ids/page `None`, title from the matching extracted source or `"Cognee"`), score 0.0, at most `k`. `get_retriever()` returns `CogneeRetriever()` when `settings.retriever == "cognee"`, else `HybridRetriever()`.

- [ ] **Step 4: Run** — expected PASS.

- [ ] **Step 5: Commit** `git add core/retrieval.py tests/test_retrieval.py && git commit -m "feat: HybridRetriever with RRF and rerank, CogneeRetriever baseline"`

---

### Task 5: Ingest writes our store first, Cognee best-effort

**Files:**
- Modify: `core/ingest.py` (all four `ingest_*`), `tests/test_ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `parse_pdf`, `text_blocks` (Task 1); `chunk` (Task 2); `index_chunks` (Task 3); existing `_upload_to_cloud`, `add_source`, `delete_source` from `core.topics`.
- Produces: `async _store(result: dict, full_text: str, blocks: list[Block]) -> None` — sets `result["status"]`, `result["error"]`, `result["source_id"]`, `result["chunk_count"]`, `result["cognee"] = {"ok": bool, "error": str | None}`, `result["excerpt"]`. All ingest results gain `chunk_count` and `cognee`.

- [ ] **Step 1: Update/write tests.**

```python
async def test_ingest_text_cloud_error(offline):  # replaces the old expectation
    # mock /api/v1/remember -> 500
    r = await ingest_text("Self-attention", title="L1", topic_id="general")
    assert r["status"] == "success" and r["chunk_count"] == 1
    assert r["cognee"]["ok"] is False and r["cognee"]["error"]

async def test_ingest_pdf_chunks_keep_pages(tmp_path):
    # 2-page PDF -> r["chunk_count"] >= 2; Chunk rows for r["source_id"] have pages {1, 2}

async def test_index_failure_rolls_back(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_dense", boom)
    r = await ingest_text("x", title="L1", topic_id="general")
    assert r["status"] == "error" and list_sources() == []
    # no Chunk rows remain
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest -q tests/test_ingest.py` — expected FAIL.

- [ ] **Step 3: Implement `_store`.** Order: `_register_in_topic(result)` → `index_chunks(result["source_id"], result["topic_id"], chunk(blocks))` via `asyncio.to_thread` → on exception `topics.delete_source(source_id)` (which purges after Task 7; until then also delete `Chunk` rows by `source_id` here) and set error `f"Indexing failed: {type(exc).__name__}: {exc}"` → `upload = await _upload_to_cloud(full_text, result["dataset_name"])` → `result["cognee"] = {"ok": upload["ok"], "error": upload.get("error")}`. Each `ingest_*` replaces its `upload ... if upload["ok"]` branch with `await _store(result, full_text, blocks)`; PDF passes `parse_pdf(file_path)`, the others `text_blocks(<raw text>)`. Keep the existing empty-input errors. Docstrings: drop "into the Cognee Cloud knowledge graph" → "into the local index (and Cognee Cloud)".

- [ ] **Step 4: Run** full suite — expected PASS (`test_full_flow`, `test_ingest_pdf_upload` unchanged).

- [ ] **Step 5: Commit** `git commit -am "feat: ingest indexes locally first; Cognee upload best-effort"`

---

### Task 6: Grounded answers with citations

**Files:**
- Modify: `core/query.py` (`ask`, `ask_scoped`), `tests/test_api.py`
- Test: `tests/test_api.py`, `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `get_retriever`, `Hit` (Task 4); `llm.complete(prompt, system=None, **kw)`.
- Produces: `async _answer(question: str, topic_id: str | None, empty_msg: str) -> dict` used by both; response `{"answer", "sources": list[str] (unique titles), "citations": [{"source_title", "page", "chunk_id", "text"}], "search_type": "HYBRID" | "COGNEE_CHUNKS", "error"}`; `ask_scoped` also keeps `topic_id`.

System prompt (exact copy):

```text
You are a study assistant. Answer only from the numbered context passages.
Cite the passage for every claim with its number in square brackets, like [2].
If the context does not contain the answer, reply exactly: This isn't in your material.
```

User prompt: each hit as `[n] {source_title}{", p." + page if page}{" — " + section if section}\n{text}`, joined by blank lines, then `\n\nQuestion: {question}`.

- [ ] **Step 1: Write failing tests.**

```python
def test_scoped_answer_cites_pages(client):
    # create topic, upload 2-page PDF with "gradient descent" on page 2 to it
    body = client.post("/ask", json={"question": "gradient descent", "mode": "scoped", "topic_id": tid}).json()
    assert body["citations"][0]["page"] == 2 and body["citations"][0]["source_title"] == "notes.pdf"

def test_ask_empty_topic_no_results(client):
    body = client.post("/ask", json={"question": "q", "mode": "scoped", "topic_id": tid}).json()
    assert body["citations"] == [] and "No results found" in body["answer"]

def test_ask_llm_down(client, monkeypatch):
    # llm.complete raises RuntimeError -> 200, body["error"] contains "RuntimeError"
```

Update `test_full_flow`: for `global` and `scoped`, assert `resp.json()["citations"][0]["source_title"] == "Lecture 1"` instead of the `"Self-attention" in answer` check (the fake LLM answers "Transformer Models"); `cross` keeps the old assertion.

- [ ] **Step 2: Run** — expected FAIL.

- [ ] **Step 3: Implement `_answer`;** `ask(question)` → `_answer(question, None, <existing no-results text>)`, `ask_scoped` → `_answer(question, topic_id, <existing topic no-results text>)`. Catch exceptions from retrieval and `llm.complete` into `error` (`f"{type(exc).__name__}: {exc}"`). Remove the now-unused GRAPH/HYBRID_COMPLETION fallback code in `ask` only if nothing else calls it (`cross_topic_ask` still uses `_recall` — keep `_recall`).

- [ ] **Step 4: Run** full suite — expected PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: scoped and global Q&A grounded on retrieved chunks with page citations"`

---

### Task 7: Real delete and move

**Files:**
- Modify: `core/topics.py` (`delete_source`, `delete_topic`, `delete_all_sources`, `move_source`), `api/main.py` (`FORGET_NOTE`), `tests/test_topics.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `delete_source_points`, `move_source_points` (Task 3), `Chunk` (Task 3).
- Produces: `_purge(s: Session, source_ids: list[int]) -> None` in `core/topics.py` (deletes Qdrant points, then `Chunk` rows, inside the caller's session before its commit). `FORGET_NOTE = "Source and its indexed chunks deleted. Data uploaded to Cognee Cloud persists."`

- [ ] **Step 1: Write failing tests** (the phase's done criterion).

```python
def test_deleted_source_never_retrieved(client):
    # ingest text "photosynthesis converts light" into tid; scoped ask cites it
    client.delete(f"/sources/{sid}")
    body = client.post("/ask", json={"question": "photosynthesis", "mode": "scoped", "topic_id": tid}).json()
    assert body["citations"] == []
    # same via DELETE /topics/{tid} and DELETE /sources (global ask) — parametrize

def test_moved_source_retrieved_only_in_new_topic(client):
    # ingest into t1, PATCH to t2: scoped ask on t1 -> no citations; on t2 -> cites it
```

- [ ] **Step 2: Run** — expected FAIL.

- [ ] **Step 3: Implement.** Call `_purge` in the three deletes with the affected source ids; `move_source` updates the source's `Chunk.topic_id` rows and calls `move_source_points`. Update the `delete_source` docstring. Remove the temporary Chunk cleanup from Task 5's rollback since `delete_source` now purges.

- [ ] **Step 4: Run** full suite — expected PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: deleting or moving a source updates Qdrant and chunk rows"`

---

### Task 8: Citation chips in the UI

**Files:**
- Modify: `ui/app.py` (Ask tab, ~lines 700–775)

**Interfaces:**
- Consumes: `citations` from `/ask` (Task 6).
- Produces: `_render_citations(citations: list[dict]) -> None` — one `st.columns` row, one `st.popover(f"[{n}] 📄 {title}" + (f" · p.{page}" if page else ""))` per citation showing the chunk text.

- [ ] **Step 1: Implement.** Call `_render_citations` for history messages and for the fresh answer; store `citations` in `chat_history`. Keep the existing "📎 Sources" expander only when there are no citations (cross mode). Change the spinner text to "🤔 Searching your material...".

- [ ] **Step 2: Verify manually.** `uvicorn api.main:app --reload` and `streamlit run ui/app.py`; ingest a multi-page PDF, ask a scoped question, confirm chips show page numbers and popovers show the text; delete the source and confirm the question no longer cites it. (First run downloads models — expect a pause.)

- [ ] **Step 3: Commit** `git commit -am "feat: citation chips with chunk popovers"`

---

### Task 9: Eval dataset and baseline runner

**Files:**
- Create: `eval/dataset.jsonl`, `eval/run.py`
- Modify: `.gitignore` (add `eval/results/`, `data/qdrant/`)

**Interfaces:**
- Consumes: `get_retriever` (Task 4), `_answer` (Task 6), `settings`.
- Produces: CLI `python -m eval.run --retriever hybrid|cognee [--embed bge-small|nomic]` writing `eval/results/<retriever>-<embed>-<YYYY-MM-DD>.jsonl`, one line per question: `{id, question, topic_id, hits: [{source_title, page, chunk_id, score}], answer, error}`.

- [ ] **Step 1: Write `eval/dataset.jsonl`** — 3 example rows with keys `{id, question, topic_id, expected_source, expected_pages, reference_answer}` (ids `q001`–`q003`, topic `general`, placeholder questions about the user's material clearly marked `"EXAMPLE — replace"` in `reference_answer`).

- [ ] **Step 2: Implement `eval/run.py`** — sets `settings.retriever`/`settings.embed_model` from args, calls `reindex()` when the target collection is empty, runs each row through `_answer`, writes results. `init_db()` first.

- [ ] **Step 3: Verify** `.venv/Scripts/python -m eval.run --retriever hybrid` produces a results file with 3 lines (needs real models + Ollama; run once locally).

- [ ] **Step 4: Commit** `git add eval .gitignore && git commit -m "feat: eval dataset scaffold and retriever baseline runner"`
