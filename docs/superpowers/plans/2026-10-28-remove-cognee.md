# Remove Cognee Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete Cognee Cloud; every feature retrieves through our retriever and generates with `llm.complete`; baseline becomes `DenseRetriever`.

**Architecture:** One shared helper in `core/query.py` turns hits into numbered context and citations; each feature passes a short retrieval query and a separate instruction. Removals follow once nothing calls them.

**Tech Stack:** unchanged (Phase 1 stack minus Cognee).

**Spec:** `docs/superpowers/specs/2026-10-28-remove-cognee-design.md`

## Global Constraints

- Tests offline; `.venv/Scripts/python -m pytest -q` green after every task.
- Retrieval query and generation prompt are always separate strings.
- Write files as UTF-8 (`encoding="utf-8"`); the repo has CRLF files.

## Review Focus

1. Topic with sources but no chunks (legacy rows) → features return their "no material" text, no exception. Test in Task 3.
2. LLM failure in cross-topic / summarize / connect → `error` set, HTTP 200. Test in Task 2.
3. Existing DB with `source.dataset_name NOT NULL` → `init_db()` drops it; new inserts work. Test in Task 3.
4. Cross-topic with no other topics → `cross_section` says none found, no crash. Test in Task 2.
5. Connect with no overlap and an LLM reply that is not in `CONCEPT | REASON` form → bullets still parsed or empty state. Covered by existing `_parse_concept_lines` fallback; test in Task 2.

---

### Task 1: DenseRetriever replaces CogneeRetriever
**Files:** `core/retrieval.py`, `eval/run.py`, `tests/test_retrieval.py`, `tests/test_eval.py`
**Produces:** `DenseRetriever.retrieve(query, topic_id, k=6) -> list[Hit]` (dense `query_points` top-k, hydrated like hybrid, score = Qdrant score); `get_retriever()` returns it for `settings.retriever == "dense"`; `eval.run --retriever hybrid|dense`.
- [ ] Tests: `test_dense_retriever_returns_hits_without_rerank` (monkeypatch `rerank` to raise; dense still returns the right page); `test_get_retriever` uses `"dense"`. Run → FAIL.
- [ ] Implement; delete `CogneeRetriever` and its test. Run → PASS. Commit.

### Task 2: Query features on our retriever
**Files:** `core/query.py`, `core/graph_utils.py`, `tests/test_api.py`, `tests/test_query.py` (new)
**Produces:** `_context(hits) -> tuple[str, list[dict]]` (numbered context, citations) used by `_answer`; rewired `cross_topic_ask`, `summarize_topic`, `connect_topics` per the spec table.
- [ ] Tests (fake retriever recording `(query, topic_id)` calls, fake `llm.complete` recording prompts):
  - `test_cross_topic_retrieves_with_question_per_topic` — calls are `(q, primary)` + `(q, other)` per other topic; one LLM call; sections split; citations present.
  - `test_cross_topic_without_other_topics`.
  - `test_summarize_retrieves_topic_text` — retrieval query == topic string; LLM prompt contains the chunk text.
  - `test_connect_falls_back_to_llm_concepts` — no entity overlap; LLM returns `CONCEPT: Gradient | REASON: r` → a shared node "Gradient".
  - `test_features_report_llm_error` — llm raises → `error` set (cross, summarize).
  Run → FAIL.
- [ ] Implement; delete `_recall`, `_extract_*_from_recall`, `_connect_via_graph_completion`'s Cognee call (replaced), `_build_text_summary`, `graph_utils.parse_search_results`, `format_source_attribution`. Run full suite → PASS. Commit.

### Task 3: Study guide, audio, Source.dataset_name
**Files:** `core/topics.py`, `core/db.py`, `tests/test_topics.py`
**Produces:** `_grounded(topic_id, instruction) -> str` (retrieves topic name scoped, k=8; generates; "" when no hits); `init_db()` drops `source.dataset_name` if present; `add_source(topic_id, source_type, title)`.
- [ ] Tests: `test_study_guide_retrieves_topic_name_not_instruction`; `test_study_guide_no_chunks_message`; `test_init_db_drops_legacy_dataset_name_column` (create old-schema table with raw SQL, `init_db()`, `add_source` works). Update `test_study_guide_is_cached` and other tests that pass `dataset_name`. Run → FAIL.
- [ ] Implement; delete `_llm_call`, `_extract_text`, `get_topic_datasets`, `datasets` key. Run → PASS. Commit.

### Task 4: Ingest, config, API, UI, README, fixtures
**Files:** `core/ingest.py`, `core/config.py`, `api/main.py`, `ui/app.py`, `README.md`, `tests/conftest.py`, `tests/test_ingest.py`, `tests/test_api.py`
- [ ] Tests: `test_health` expects only `llm`; ingest result has no `cognee`/`dataset_name`; fixture handler allows only `/api/tags`; remove `test_ingest_text_cloud_error`, dataset-name tests. Run → FAIL.
- [ ] Implement removals; UI health row, spinners and caption text without Cognee; README rewritten for the local stack. `grep -ri cognee core api ui eval tests README.md` → only the spec/plan history mentions. Run → PASS. Commit.
