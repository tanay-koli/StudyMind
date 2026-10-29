# Phase 2 — Grounded Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Query rewriting (multi / HyDE behind a flag), cross-topic decomposition, a map-reduce `StudyGuide` that covers every section, and topic connections computed from per-chunk concept triples extracted in the background.

**Architecture:** A `RewritingRetriever` wraps the existing retriever. `core/studyguide.py` builds section units from `Chunk` rows, maps each to JSON, and reduces. `core/concepts.py` owns two new tables, a background extraction worker started in the API lifespan, and the overlap query that `connect_topics` uses instead of an LLM call.

**Tech Stack:** Python 3.13, SQLModel/SQLite, Qdrant embedded, fastembed, LiteLLM → Ollama `qwen2.5:7b`, Pydantic v2, FastAPI, Streamlit, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-28-phase2-grounded-generation-design.md`

## Global Constraints

- All tests offline: fake `llm.complete`, fake embedders/reranker, `:memory:` Qdrant (see `tests/conftest.py`). No network, no model downloads.
- Storage changes are new tables only; `create_all()` adds them. No `ALTER`.
- Write files as UTF-8 explicitly; no shell heredocs with `\n` escapes.
- Run tests with `.venv/Scripts/python -m pytest -q`. Baseline: 81 pass.
- `QUERY_REWRITE` default is `none`; behaviour with the default must match Phase 1.
- `num_ctx=8192` for `ollama/*` models.
- Study guide unit budget 2,500 tokens; map retry once; UI timeout for the study guide 900 s.
- Concept batch size 4; at most 5 triples per chunk; triple parts ≤ 60 chars; connect k = 15; top 8 shared concepts.
- Pending note copy, exactly: `N of M passages are still being processed — try again shortly.`
- Failed-section key point copy, exactly: `(could not summarise this section)`
- Every limitation taken knowingly gets a `# shortcut: <limit>, <when to upgrade>` comment.

## Review Focus

1. **Ollama down while the worker runs:** every batch fails. The worker must finish the pass and go back to waiting, not spin retrying the same chunks. Test in Task 5.
2. **Ollama down during a study guide:** every map unit fails. Return the "Could not generate" message (not a guide made only of failed sections) and cache nothing. Test in Task 4.
3. **LLM wraps JSON in ```json fences or adds prose around it:** `complete_json` must still parse the first JSON object. Test in Task 1.
4. **Topic name appears in the concepts with different case/plural** (`"Neural Networks"` vs `"neural network"`): it must not be listed as a shared concept. Test in Task 7.
5. **Paraphrase / sub-question replies with numbering, blanks, duplicates, or the original question repeated:** they are cleaned and de-duplicated; an unusable reply falls back to the original. Tests in Tasks 2 and 3.

---

### Task 1: LLM context window and JSON helper

**Files:**
- Modify: `core/llm.py`
- Test: `tests/test_llm.py` (create)

**Interfaces:**
- Produces: `LLMClient.complete(...)` now sets `num_ctx=8192` for `ollama/*` models unless passed. `async def LLMClient.complete_json(self, prompt: str, system: str | None = None, **kwargs) -> dict` — calls `self.complete(..., response_format={"type": "json_object"})`, returns the first JSON object in the reply, raises `ValueError` when none parses.

- [ ] **Step 1: Write the failing tests**

```python
async def test_complete_sets_num_ctx_for_ollama(monkeypatch):
    seen = {}
    async def fake_acompletion(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))], usage=None)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    await LLMClient().complete("q")
    assert seen["num_ctx"] == 8192
    await LLMClient().complete("q", num_ctx=2048)
    assert seen["num_ctx"] == 2048

@pytest.mark.parametrize("reply", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} hope that helps'])
async def test_complete_json_extracts_object(monkeypatch, reply): ...   # == {"a": 1}

async def test_complete_json_raises_on_no_object(monkeypatch): ...     # "no json here" -> ValueError
```

Use a fresh `LLMClient()` with `complete` monkeypatched on that instance for the JSON tests (the conftest fakes `llm.complete` on the shared instance only).

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_llm.py -q` — expect FAIL.
- [ ] **Step 3: Implement.** `kwargs.setdefault("num_ctx", 8192)` inside the existing `ollama/` branch. `complete_json`: `json.JSONDecoder().raw_decode` from each `{` position until one decodes to a dict.
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Commit** `feat: num_ctx for Ollama and a JSON completion helper`

---

### Task 2: Query rewriting

**Files:**
- Create: `core/rewrite.py`
- Modify: `core/retrieval.py` (`Retriever` protocol, both retrievers, `get_retriever`), `core/config.py`, `eval/run.py`
- Test: `tests/test_rewrite.py` (create), `tests/test_eval.py`

**Interfaces:**
- Consumes: `retrieval.rrf`, `retrieval.Hit`, `llm.complete`.
- Produces:
  - `Settings.query_rewrite: str = "none"  # none | multi | hyde`
  - `retrieve(self, query: str, topic_id: Optional[str], k: int = 6, rerank_query: Optional[str] = None) -> list[Hit]` on `HybridRetriever` (reranks with `rerank_query or query`) and `DenseRetriever` (accepts and ignores it).
  - `class RewritingRetriever: __init__(self, base, mode: str)`, same `retrieve` signature.
  - `async def paraphrases(question: str) -> list[str]` — up to 3, cleaned (numbering/bullets stripped, blanks dropped, case-insensitive de-dupe, original removed); `[]` on LLM failure.
  - `async def hypothetical_answer(question: str) -> str` — `""` on failure.
  - `get_retriever()` returns `RewritingRetriever(base, settings.query_rewrite)` when it is not `"none"` (import `core.rewrite` inside the function to avoid a cycle).
  - `eval/run.py`: `--rewrite {none,multi,hyde}` (default `settings.query_rewrite`); output file name becomes `<retriever>-<embed>-<rewrite>-<date>.jsonl`.

- [ ] **Step 1: Write the failing tests** (a `RecordingRetriever` fake returning hits keyed by query and recording `(query, topic_id, k, rerank_query)`):
  - `test_multi_fuses_original_and_paraphrases`: LLM reply `"1. what is X?\n\n- What is X?\nexplain Y\nexplain Z\nextra"` → base called with the original plus `["what is X?", "explain Y", "explain Z"]`; a chunk ranked first in two lists comes first in the result; `len(result) <= k`.
  - `test_multi_drops_original_echo`: a reply line equal to the question (any case) is not retrieved twice.
  - `test_hyde_retrieves_with_answer_reranks_with_question`: base called once with `query=<hypothetical text>`, `rerank_query=<question>`.
  - `test_rewrite_falls_back_when_llm_fails` (parametrize mode): LLM raises → base called exactly once with the original question.
  - `test_hybrid_reranks_with_rerank_query`: with real `HybridRetriever` on indexed chunks, `rerank_query` decides the order (conftest's word-overlap `fake_rerank`).
  - `test_get_retriever_wraps_when_flag_set` (monkeypatch `settings.query_rewrite`).
  - In `tests/test_eval.py`: the output name contains the rewrite mode.
- [ ] **Step 2: Run** `pytest tests/test_rewrite.py tests/test_eval.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces. In multi mode each query retrieves with `k=k`; fuse the `chunk_id` rankings with `rrf`, map back to `Hit`s (first occurrence), take `[:k]`. Prompts: paraphrases — "Rewrite this question 3 different ways for searching study notes. One per line, nothing else."; HyDE — "Write a short passage (3-4 sentences) from study notes that would answer this question."
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Commit** `feat: multi-query and HyDE rewriting behind QUERY_REWRITE`

---

### Task 3: Cross-topic decomposition

**Files:**
- Modify: `core/query.py` (`cross_topic_ask`, new helpers)
- Test: `tests/test_query.py`

**Interfaces:**
- Produces:
  - `async def _sub_questions(question: str) -> list[str]` — one LLM call; keep only lines ending in `?` after stripping numbering/bullets/bold; de-dupe; at most 3; returns `[question]` when fewer than 2 remain or the LLM fails.
  - `def _topics_with_chunks(topic_ids: list[str]) -> set[str]` — one `SELECT DISTINCT Chunk.topic_id ... WHERE topic_id IN (...)`.
- `cross_topic_ask` keeps its signature and result keys. Per topic, retrieve every sub-question with the existing k (6 primary, 2 others), merge in sub-question order de-duplicated by `chunk_id`, cap at that k. Topics not in `_topics_with_chunks` are not retrieved. The synthesis prompt keeps using the **original** question.

- [ ] **Step 1: Update and add tests.** In the `fake` fixture monkeypatch `query._topics_with_chunks` to `lambda ids: set(ids)`. Existing cross-topic tests now see a decomposition call first: assert on `prompts[-1]` for synthesis. New tests:
  - `test_cross_topic_decomposes_and_retrieves_each_sub_question`: reply sequence `["1. What is A?\n2. What is B?", "<synthesis>"]` → retriever calls are each sub-question × each topic; synthesis prompt contains the original question.
  - `test_cross_topic_skips_topics_without_chunks`: `_topics_with_chunks` returns only the primary → no retrieve call for the other topic.
  - `test_sub_questions_cleanup` (parametrize): `"**1. What is A?**\n\nWhat is A?\nnot a question"` → `["What is A?"]` → falls back to `[question]`; LLM raises → `[question]`.
  - `test_topics_with_chunks` against the real DB via `ingest_text`.
  
  Extend the fixture so `reply` may be a list consumed in order.
- [ ] **Step 2: Run** `pytest tests/test_query.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Commit** `feat: cross-topic decomposes into sub-questions and skips empty topics`

---

### Task 4: Study guide map-reduce

**Files:**
- Create: `core/studyguide.py`
- Modify: `core/topics.py` (`generate_study_guide`), `api/main.py` (study-guide route), `ui/app.py` (`api()` timeout parameter, study-guide call)
- Test: `tests/test_studyguide.py` (create), `tests/test_topics.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `llm.complete_json` (Task 1), `chunking.token_spans` (call through the module so the conftest fake applies), `Chunk`, `Source`.
- Produces (in `core/studyguide.py`):
  - Pydantic models `Term`, `SectionSummary`, `StudyGuide` exactly as in the spec.
  - `UNIT_TOKENS = 2500`, `FAILED_POINT = "(could not summarise this section)"`.
  - `@dataclass class Unit: source_title: str; headings: list[str]; pages: str; text: str` — `pages` is `"3"`, `"3-5"`, or `""` when no page.
  - `def units(topic_id: str) -> list[Unit]` — algorithm below.
  - `async def build(topic_name: str, unit_list: list[Unit]) -> tuple[StudyGuide, bool]` — the bool is `complete` (no map unit failed and the reduce succeeded).
  - `def render_markdown(guide: StudyGuide) -> str` — `# Study Guide: <topic>`, overview, `## Sections` with `### <heading> (<source> · p.<pages>)` and bullets, `## Key Concepts`, `## Terms`, `## Review Questions`.
- `topics.generate_study_guide(topic_id, force=False) -> dict` with keys `guide: str` and `data: dict | None`. Cache: store `StudyGuide.model_dump_json()`; a cached value failing `StudyGuide.model_validate_json` is regenerated. Cache only when `complete`. When every map unit failed, return `{"guide": <existing "Could not generate" message>, "data": None}`.
- `POST /topics/{id}/study-guide` returns that dict unchanged.
- `ui/app.py`: `api(method, path, timeout=300, **kwargs)`; the study-guide call passes `timeout=900` and reads `["guide"]` as today.

`units` algorithm (the signature doesn't determine this):

```
rows = chunks of topic joined with Source.title, ordered by (source_id, ord)
groups = consecutive rows with equal (source_id, section_heading)
for each source: walk its groups, appending to the current unit while
    tokens(current) + tokens(group) <= UNIT_TOKENS; otherwise start a new unit.
    A group bigger than UNIT_TOKENS on its own is split by chunk into budget-sized units.
headings = distinct non-None headings in the unit, in order; a unit with none gets
    "<source_title> (part n)"; SectionSummary.heading = " / ".join(headings)
```

Map prompt asks for `{"key_points": [...], "terms": [{"term": ..., "definition": ...}]}`. Validate with a small Pydantic model; on `ValueError`/`ValidationError` retry once, then the section gets `key_points=[FAILED_POINT]`. Reduce sees only the joined key points and terms and returns `{"overview", "key_concepts", "review_questions"}`; on failure those stay empty. `sections` = one `SectionSummary` per unit in unit order, never taken from the reduce reply. `terms` = map terms de-duplicated by lowercase `term`.

- [ ] **Step 1: Write the failing tests** (seed chunks via `ingest_text` with Markdown-free text plus direct `Chunk` inserts where headings/pages matter):
  - `test_units_group_by_section_and_pack`: three small sections in one source → one unit with all three headings; with the budget monkeypatched to small, they split at section boundaries.
  - `test_units_split_oversized_section`: one section over budget → several units, each ≤ budget (whitespace tokens).
  - `test_units_headingless_windows`: chunks with no heading → heading `"<title> (part 1)"`.
  - `test_every_section_present_even_if_reduce_omits_it`: map replies valid JSON; reduce reply mentions nothing → `[s.heading for s in guide.sections]` equals the unit headings; `complete is True`.
  - `test_map_retries_once_then_marks_section`: first unit's map reply bad twice → its key points `== [FAILED_POINT]`, `complete is False`, exactly 2 map attempts for it.
  - `test_render_markdown_contains_sections`.
  - In `tests/test_topics.py`: drop `generate_study_guide` from the `test_generation_retrieves_topic_name_not_instruction` parametrize; rewrite `test_study_guide_is_cached` for the dict result and JSON cache; add `test_incomplete_guide_not_cached`, `test_all_units_failing_returns_could_not_generate` (fake LLM raises → `"Could not generate" in res["guide"]`, `data is None`, nothing cached), `test_legacy_markdown_cache_regenerated`.
  - In `tests/test_api.py`: the route returns `guide` and `data`.
- [ ] **Step 2: Run** `pytest tests/test_studyguide.py tests/test_topics.py tests/test_api.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces and algorithm.
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Commit** `feat: study guide map-reduce over every section, returned as StudyGuide`

---

### Task 5: Concept tables and extraction

**Files:**
- Modify: `core/db.py` (two tables), `core/topics.py` (`_purge`, `move_source`)
- Create: `core/concepts.py`
- Test: `tests/test_concepts.py` (create), `tests/test_topics.py`

**Interfaces:**
- Consumes: `llm.complete_json` (Task 1).
- Produces:
  - `ConceptEdge`, `ConceptExtraction` in `core/db.py`, exactly as in the spec.
  - `BATCH = 4`, `MAX_TRIPLES = 5`, `MAX_PART = 60`.
  - `def normalise(name: str) -> str` — rules as in the spec, with `# shortcut: no lemmatisation or synonyms, upgrade if topic maps look fragmented`.
  - `async def extract_pending() -> int` — processes every pending chunk in batches of `BATCH` and returns how many chunks were marked done. Failed batches are skipped for the rest of this call (an in-memory set), so a pass always ends.
  - `_purge` also deletes `ConceptEdge` / `ConceptExtraction` rows whose `chunk_id` belongs to the purged sources (select the chunk ids before deleting the chunks). `move_source` also updates `ConceptEdge.topic_id` for the source.

The prompt numbers the batch's chunks 1..n and asks for `{"chunks": [{"n": 1, "triples": [["head", "relation", "tail"], ...]}]}`, with concepts as short noun phrases. Drop a triple unless it has 3 non-empty string parts, each ≤ `MAX_PART`, and `normalise(head) != normalise(tail)`. Keep the first `MAX_TRIPLES` per chunk. Write edges plus a `ConceptExtraction` row for every chunk in the batch (0 triples included) in one session/commit, skipping chunk ids whose `Chunk` row no longer exists. Add `# shortcut: a chunk that always fails is retried on every wake/restart, add an attempt count if logs show repeats`.

- [ ] **Step 1: Write the failing tests:**
  - `test_normalise` (parametrize): `"The Neural Networks"→"neural network"`, `"TCP/IP"→"tcpip"` (punctuation removed), `"  Gradient   Descent "→"gradient descent"`, `"analysis"→"analysis"`, `"class"→"class"`, `"virus"→"virus"`, `"an Enzyme"→"enzyme"`.
  - `test_extract_stores_edges_and_done_rows`: 5 chunks, fake JSON replies → 2 LLM calls (4 + 1); edges have the right `chunk_id`/`source_id`/`topic_id`/`head_norm`; every chunk has a done row; invalid triples dropped; a 6th triple dropped.
  - `test_failed_batch_stays_pending_and_pass_ends`: LLM always returns non-JSON → `extract_pending()` returns `0`, makes exactly `ceil(n/4)` calls, and no done rows exist (Review Focus 1).
  - `test_deleted_chunk_skipped`: delete the source inside the fake LLM call → no edges/done rows for its chunks, no error.
  - In `tests/test_topics.py`: `test_delete_source_purges_concepts`, `test_move_source_moves_concepts`.
- [ ] **Step 2: Run** `pytest tests/test_concepts.py tests/test_topics.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Commit** `feat: concept_edge table and batched triple extraction`

---

### Task 6: Background worker lifecycle

**Files:**
- Modify: `core/concepts.py`, `api/main.py` (lifespan), `core/ingest.py` (`_store`)
- Test: `tests/test_concepts.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces:
  - `def start() -> asyncio.Task` — creates the module's `asyncio.Event` and a task running: `extract_pending()` → `await event.wait()` → `event.clear()` → repeat. Any exception in a pass is logged and the loop continues.
  - `def wake() -> None` — sets the event; no-op when `start()` has not run (tests, eval).
  - `async def stop() -> None` — cancels the task, awaits it, resets the module state.
  - Lifespan: `concepts.start()` after `ensure_indexed`; `await concepts.stop()` first on shutdown.
  - `_store` calls `concepts.wake()` after a successful index.

- [ ] **Step 1: Write the failing tests:**
  - `test_worker_wakes_on_ingest`: `start()`, ingest text, wait (poll with `asyncio.sleep(0)` up to a bound) until done rows exist; `await stop()`.
  - `test_wake_without_start_is_noop`.
  - `test_stop_cancels_cleanly`: `start()` then `stop()` while waiting → no exception, a second `stop()` is a no-op.
  - The existing `tests/test_api.py` TestClient flows must still pass with the worker running (the fake LLM replies non-JSON, so batches fail and the worker waits).
- [ ] **Step 2: Run** `pytest tests/test_concepts.py tests/test_ingest.py tests/test_api.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the full suite — expect all pass, with no "Task was destroyed but it is pending" warnings.
- [ ] **Step 5: Commit** `feat: background concept extraction worker, woken by ingest`

---

### Task 7: Connections from concept overlap

**Files:**
- Modify: `core/concepts.py`, `core/query.py` (`connect_topics`; delete `_connect_via_llm`, `_parse_concept_lines`)
- Test: `tests/test_concepts.py`, `tests/test_query.py` (delete the three LLM-connect tests and the `_parse_concept_lines` test)

**Interfaces:**
- Consumes: `ConceptEdge`, `ConceptExtraction`, `normalise` (Task 5); existing `render_pyvis_graph`, `_node_id`, `SOURCE_COLORS`, `_context`-style citation dicts.
- Produces:
  - `CONNECT_K = 15`, `TOP_SHARED = 8`, `PENDING_NOTE = "{n} of {m} passages are still being processed — try again shortly."`
  - `@dataclass class Shared: name: str; support: int; edge_a: ConceptEdge; edge_b: ConceptEdge`
  - `def shared_concepts(chunks_a: list[int], chunks_b: list[int], exclude: set[str]) -> list[Shared]` — a concept belongs to a side if it is the `head_norm` or `tail_norm` of an edge on that side's chunks. Shared = intersection minus `exclude`. `support` = distinct chunks mentioning it across both sides. Sort by `(-support, name)`, take `TOP_SHARED`. `name` = the display form of the first edge part on side A.
  - `def pending_count(chunk_ids: list[int]) -> int` — ids without a `ConceptExtraction` row.
  - `connect_topics(topic_a, topic_b)` keeps its result keys. Retrieve both with `k=CONNECT_K`, `exclude={normalise(topic_a), normalise(topic_b)}`. Graph: topic A → concept → topic B, each edge labelled with that side's `edge.relation`. Summary: one line per concept, `**<name>** — A: <head> <relation> <tail> (📄 <title> · p.<page>); B: ...`. Append `PENDING_NOTE` when `pending_count(distinct retrieved ids) > 0`. `connection_method = "concept_overlap"`; no shared concepts → the existing empty-state summary (plus the pending note if any) and `connection_method = "none"`. No `llm.complete` call.

- [ ] **Step 1: Write the failing tests** (plant `ConceptEdge` / `ConceptExtraction` rows for chunks created via `ingest_text`; a fake retriever returns those chunks per topic):
  - `test_shared_concepts_overlap_and_rank`: concept on 3 chunks beats one on 2; ties alphabetical; at most 8.
  - `test_topic_names_excluded`: topic `"Neural Networks"`, edges containing `"neural network"` → not shared (Review Focus 4).
  - `test_connect_builds_graph_without_llm`: `connection_method == "concept_overlap"`, a shared node exists, edges carry the planted relations, fake LLM never called.
  - `test_connect_pending_note`: one retrieved chunk without a done row → the summary contains `"1 of 2 passages are still being processed — try again shortly."`
  - `test_connect_no_overlap_empty_state`: `connection_method == "none"`, `nodes == []`.
- [ ] **Step 2: Run** `pytest tests/test_concepts.py tests/test_query.py -q` — expect FAIL.
- [ ] **Step 3: Implement** per the Interfaces; delete the old LLM-connect code.
- [ ] **Step 4: Run** the full suite — expect all pass.
- [ ] **Step 5: Real-model check** (stop uvicorn first if it is running): start the API with Ollama up, ingest one PDF, wait for the log to show the extraction pass finishing, then check:
  - `/connect` on two related topics returns `concept_overlap` with real shared concepts;
  - the study guide lists every section heading of the PDF;
  - `python -m eval.run --retriever hybrid --rewrite multi` and `--rewrite hyde` both write results.
  
  Record what happened in the commit message body.
- [ ] **Step 6: Commit** `feat: topic connections from concept overlap; LLM connect removed`
