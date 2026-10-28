# Phase 1 — Own ingestion and hybrid retrieval (design)

Source: "StudyMind v2 — Implementation Plan.pdf", Phase 1.

## Goal

Build our own RAG core behind a `Retriever` interface with two implementations:
`CogneeRetriever` (today's behaviour, kept as the eval baseline) and `HybridRetriever` (ours).

**Done when:** scoped Q&A answers cite page numbers, and a deleted source never appears in results again.

## Decisions

| Area | Choice | Why |
|---|---|---|
| Embeddings | fastembed (ONNX, CPU): `BAAI/bge-small-en-v1.5` (384-d) primary, `nomic-ai/nomic-embed-text-v1.5` (768-d) second | Local and free; replaces the plan's paid `text-embedding-3-small`. No torch. |
| Sparse | fastembed `Qdrant/bm25` sparse vectors stored in Qdrant | Same filters for dense and sparse; delete/move stay consistent. |
| Fusion | Our own RRF in Python, k = 60 | Explicit, testable constant. |
| Reranker | fastembed `TextCrossEncoder("BAAI/bge-reranker-base")`, fused top 30 → 6 | Named in the plan; same dependency. |
| Vector store | `qdrant-client` embedded mode at `data/qdrant` (`:memory:` in tests) | No Docker until Phase 7. |
| Parsing | PyMuPDF `get_text("dict")` | Font sizes give section headings; Docling not needed yet. |
| Token counting | The bge tokenizer (`tokenizers`, shipped with fastembed) | 512 tiktoken tokens would exceed bge's 512-token window and be silently truncated. |

New dependency: `qdrant-client[fastembed]` (pinned).

## Storage

New SQLModel table `Chunk`:
`id (int pk), source_id (fk source.id, index), topic_id (index), chunker, ord, page (nullable), section_heading (nullable), text`.

Qdrant: one collection per embedding model (`chunks_bge_small`, `chunks_nomic`), created on first use.
Each point: id = `Chunk.id`, named dense vector `dense`, sparse vector `bm25`, payload `{source_id, topic_id}` (payload indexes on both).

New settings (pydantic-settings, `.env`):
- `EMBED_MODEL=bge-small` (`bge-small` | `nomic`)
- `CHUNKER=structure` (`structure` | `fixed`)
- `RETRIEVER=hybrid` (`hybrid` | `cognee`)
- `QDRANT_PATH=data/qdrant`

## Modules

### `core/parse.py`
`parse_pdf(path) -> list[Block]` where `Block = {page, section_heading, text}`.
A line is a heading when its font size is ≥ 1.2 × the document's median body size, or it is short (≤ 12 words) and bold.
The current heading carries forward to following blocks. Non-PDF sources (YouTube, URL, text) become one block with `page=None`.

### `core/chunking.py`
- `fixed(blocks, size=512, overlap=64)`: sliding window over the concatenated token stream; each chunk takes `page` / `section_heading` of the block where it starts.
- `structure(blocks, max_tokens=512)`: group blocks by section heading, then split groups larger than `max_tokens` by paragraph, falling back to the fixed window.
Both return `list[ChunkDraft] = {page, section_heading, text}` and never emit empty chunks.

### `core/retrieval.py`
- `Hit` dataclass: `chunk_id, source_id, source_title, page, section, text, score`.
- `Retriever` protocol: `async retrieve(query: str, topic_id: str | None, k: int = 6) -> list[Hit]`.
- `HybridRetriever`: dense top 30 and sparse top 30 (topic filter when given) run concurrently, `rrf(rankings, k=60)`, cross-encoder rerank to `k`, then hydrate from `Chunk` + `Source` rows.
- `CogneeRetriever`: wraps the existing `_recall` (CHUNKS search, `datasets` filter); hits have no page.
- `get_retriever()` picks by `settings.retriever`.
- Store functions: `index_chunks(source, drafts)`, `delete_source_points(source_ids)`, `move_source_points(source_id, topic_id)`.
- fastembed calls are sync; they run in `asyncio.to_thread`. Models load lazily once per process.

## Changed flows

**Ingest** (`core/ingest.py`, all four sources): parse → chunk → embed → write `Chunk` rows + Qdrant points.
Our store is primary: the ingest succeeds when local indexing succeeds.
The Cognee upload stays (baseline) but becomes best-effort, reported as `result["cognee"] = {"ok": bool, "error": str | None}`.
Result adds `chunk_count`.

**Ask**: `ask_scoped` and global `ask` call `get_retriever().retrieve(...)`, then `llm.complete` with numbered context blocks and a system prompt that requires an inline `[n]` per claim and an explicit "This isn't in your material." when context is missing.
Response keeps `answer`, `sources`, `error` and adds `citations: [{source_title, page, chunk_id, text}]` (`text` feeds the UI popover).
Cross-topic, summarize, connect, study guide and audio stay on Cognee until Phase 2.

**Delete / move** (`core/topics.py`):
- `delete_source`, `delete_topic`, `delete_all_sources` also delete Qdrant points (by `source_id` filter) and `Chunk` rows.
- `move_source` also updates `topic_id` on the source's Qdrant payloads and `Chunk` rows; otherwise moved chunks would keep answering in the old topic.
- `FORGET_NOTE` in the API is updated: local data is deleted, Cognee Cloud data still is not.

**UI** (`ui/app.py`): under each answer, citation chips `📄 Title · p.12` (page omitted when `None`); each chip is an `st.popover` showing the chunk text.

## Eval

- `eval/dataset.jsonl`: one object per line `{id, question, topic_id, expected_source, expected_pages, reference_answer}`; 3 example rows. The user adds ~15 real questions per week.
- `eval/run.py --retriever hybrid|cognee`: runs every question through the retriever and the answer step, writes `eval/results/<retriever>-<date>.jsonl` with hits and answers. Metrics are Phase 6.

## Testing (offline)

`conftest.py` adds: Qdrant `:memory:`, fake dense / sparse embedders (deterministic hash vectors) and a fake reranker (keeps fused order), so no model downloads.
New tests:
- parser finds headings and pages in a PDF generated with PyMuPDF
- both chunkers: token limits, overlap, metadata on every chunk, no empty chunks
- `rrf` scores and ordering
- scoped ask returns citations with page numbers
- deleted source is never retrieved again (also via topic delete and delete-all)
- moved source is retrieved only under its new topic
- ingest succeeds when the Cognee upload fails
Existing 28 tests keep passing.

## Out of scope

Docling, side-by-side chunker/embedder comparison in one collection, fastembed-gpu, hosted rerankers, Qdrant server/Docker, retrieval metrics, moving cross-topic / study guide / audio off Cognee.

## Risks

- First real run downloads ~400 MB of ONNX models (bge-small, nomic, bm25, bge-reranker-base).
- Embedded Qdrant holds a file lock: one process at a time (fine for `uvicorn --reload`; not for multiple workers).
- bge-reranker-base on CPU: ~1–2 s per question for 30 candidates.
