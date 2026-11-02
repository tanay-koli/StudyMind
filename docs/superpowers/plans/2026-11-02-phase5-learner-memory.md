# Phase 5 — Learner Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quizzes and reviews in the Agent chat, graded and scheduled per concept with SM-2. Learner facts are remembered and fed into prompts. Done when planted wrong answers are exactly what the next review targets.

**Architecture:** `core/tutor.py` holds the quiz logic: target picking, question generation, grading, SM-2 and recording. `core/memory.py` holds the learner facts. Concepts per chunk come from the existing `concept_edge` triples, through two helpers in `core/concepts.py`. The agent gains a `quiz` state that survives across turns, plus the nodes `quiz_start`, `quiz_answer`, `quiz_stop` and `remember`. `GET /learner` feeds a sidebar progress panel.

**Tech Stack:** SQLModel, Pydantic, LangGraph, LiteLLM → Ollama qwen2.5:7b, FastAPI, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-phase5-learner-memory-design.md`

## Global Constraints

- Everything local; tests offline with the conftest fakes. Async tests use `@pytest.mark.asyncio`. Fake `llm.complete` / `llm.complete_json` take `(prompt, system=None, **kw)`.
- New tables only. `QuizAttempt` and `LearnerFact` use `__table_args__ = {"sqlite_autoincrement": True}`.
- Dates: `date.today()`, stored as ISO strings. Code takes `today` as a parameter wherever a test needs to pin it (`sm2`, `record`, `pick_targets`).
- Constants: `SESSION_SIZE = 5`, `MAX_FACTS = 20`, `QUALITY = {"correct": 5, "partial": 3, "wrong": 1}`, `MIN_EASE = 1.3`.
- Review order: `(next_review, ease, -wrong, concept)`.
- The exact user-facing strings are in the spec's Agent section. Copy them verbatim.
- Edit files with the Edit tool or Python string replacement, not shell heredocs (heredocs mangled `\n` / `\b` twice before).
- The suite currently has 197 passing tests and takes 1.5–2.5 min. Run it in the background with the output to a log, and read the last line.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A quiz answered after the material was deleted:** the current question's chunk is gone, and `grade` must still work. Grading uses the question's stored citation text, not a fresh chunk lookup. Test in Task 4.
2. **An answer like "I think it's B because…" to a multiple-choice question:** the first-letter rule would read "I". The rule must take a standalone letter A–D (`\b[A-D]\b` after stripping `()` and `.`) before falling back to option text. Test in Task 4.
3. **Learner facts from a quiz answer** ("my answer is B"): never stored, because `remember` skips `quiz_answer` turns. Test in Task 6.
4. **Review with a topic selected while the weak concepts live in another topic:** the review returns nothing for this topic and says so, not concepts from elsewhere. Test in Task 3.
5. **The same concept appears in two quiz targets** (two chunks share it): the targets are distinct by concept. Test in Task 3.

---

### Task 1: Tables, SM-2 and record

**Files:** Modify `core/db.py`. Create `core/tutor.py`. Test `tests/test_tutor.py`.

**Interfaces — Produces:**
- `QuizAttempt`, `ConceptMastery`, `LearnerFact` (spec Storage, verbatim).
- `tutor.sm2(row: ConceptMastery, verdict: str, today: date) -> ConceptMastery` (mutates and returns).
- `tutor.record(question: Question, answer: str, verdict: Optional[str], feedback: str, topic_id: Optional[str], today: Optional[date] = None) -> None`.
- `tutor.Question` (spec, Pydantic).

- [ ] **Step 1: Failing tests**
  - `test_sm2_correct_three_times`: intervals `[1, 6, 16]`, eases `[2.6, 2.7, 2.8]` (approximately), `next_review` = today + interval.
  - `test_sm2_partial_and_wrong`: from a fresh row, partial → `interval_days == 1`, `ease == approx(2.36)`. A fresh wrong → `ease == approx(1.96)` and `repetitions == 0`. Five wrongs → `ease == 1.3` (the floor).
  - `test_record_writes_attempt_and_mastery`: correct → a `QuizAttempt` row plus mastery with `attempts == 1` and `wrong == 0`. A later wrong → `attempts == 2`, `wrong == 1`, `last_verdict == "wrong"`. `verdict=None` → the attempt is stored and mastery is unchanged.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest -q tests/test_tutor.py`. Expected: ImportError.
- [ ] **Step 3: Implement** the tables and these three pieces of `tutor.py`.
- [ ] **Step 4: Full suite** (background). Expected: all pass.
- [ ] **Step 5: Commit** `feat: quiz attempts, concept mastery with SM-2, learner fact table`

---

### Task 2: Concepts per chunk

**Files:** Modify `core/concepts.py`. Test `tests/test_concepts.py`.

**Interfaces — Produces:** `chunk_concepts(chunk_ids: list[int]) -> dict[int, list[str]]`; `chunks_for_concept(concept: str, topic_id: Optional[str] = None) -> list[int]`.

- [ ] **Step 1: Failing tests.** Reuse or copy the `_planted` helper from `tests/test_query.py`; move it into `tests/conftest.py`-style shared code only if both files need it.
  - `test_chunk_concepts_top_three_by_frequency`: chunk 0 has edges (`Enzyme`, `Substrate`), (`Enzyme`, `Active site`), (`Enzymes`, `Inhibitor`), (`Rate`, `Temperature`) → `["enzyme", "substrate", "active site"]`. A chunk without edges is absent from the result.
  - `test_chunks_for_concept_orders_by_mentions_and_topic`: a concept on 2 chunks (3 mentions against 1) → `[c3, c1]`. Filtering by another topic → `[]`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** One query each. Count `head_norm` and `tail_norm` per chunk in Python, with first-appearance tie-breaking via `Counter` insertion order.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: concepts per chunk from the stored triples`

---

### Task 3: `pick_targets`

**Files:** Modify `core/tutor.py`. Test `tests/test_tutor.py`.

**Interfaces — Consumes:** Task 2 helpers; `get_retriever(rewrite=False)`; `retrieval.relevant`. **Produces:** `pick_targets(topic_id: Optional[str], about: Optional[str] = None, review: bool = False, n: int = SESSION_SIZE, today: Optional[date] = None) -> list[tuple[str, int]]`.

- [ ] **Step 1: Failing tests**
  - `test_quiz_targets_from_retrieved_chunks`:
    - plant 3 chunks with concepts and real text, index them with `retrieval.index_chunks` so retrieval finds them, then plant their edges;
    - `pick_targets(tid, about="enzyme")` → concepts distinct, each paired with a chunk that mentions it, at most `n` (Review Focus 5: two chunks share `enzyme`, and it appears once).
  - `test_review_targets_only_weak_or_due_in_order`: mastery rows A (wrong, ease 1.96), B (partial, 2.36), C (correct, next_review tomorrow), D (correct, next_review = today, so due) → `[A, B, D]`, each with a chunk.
  - `test_review_respects_topic` (Review Focus 4): weak concepts whose chunks are only in topic X → `pick_targets(Y, review=True) == []`.
  - `test_review_empty_when_nothing_due`: all correct and not due → `[]`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** per the spec. Quiz query: `about or topic name or "overview"`; topic name via `query._topic_name`.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: pick quiz targets from material and review targets from mastery`

---

### Task 4: `generate_question` and `grade`

**Files:** Modify `core/tutor.py`. Test `tests/test_tutor.py`.

**Interfaces — Produces:**
- `generate_question(concept: str, chunk_id: int, kind: str, facts: str = "") -> Optional[Question]`.
- `grade(question: Question, answer: str, facts: str = "") -> tuple[Optional[str], str]`.
- `_mcq_letter(answer: str, options: list[str]) -> Optional[str]`.

- [ ] **Step 1: Failing tests**
  - `test_generate_valid_mcq_and_short`: fake JSON → a `Question` with `chunk_id`, `concept` and a `citation` whose `chunk_id` matches; `facts` appears in the system prompt.
  - `test_generate_rejects_bad_mcq_twice`: 3 options twice → `None`, with 2 calls.
  - `test_mcq_letter_forms` (Review Focus 2):
    - `"b"`, `"(B)"`, `"B."`, `"I think it's B because..."` → `"B"`;
    - the full text of option C → `"C"`;
    - `"no idea"` → None.
  - `test_grade_mcq_correct_makes_no_call` / `test_grade_mcq_wrong_explains_once`.
  - `test_grade_short_verdicts`: the fake rubric gives each verdict → passed through. Invalid twice → `(None, "Couldn't grade this answer.")`.
  - `test_grade_uses_stored_citation_text` (Review Focus 1): delete the chunk row, then grade a short answer → the rubric prompt still contains the citation text.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `_mcq_letter`: strip `()` and `.`, then search `\b([A-Da-d])\b`. Accept a single letter, or a letter after "is"/"it's" — use the last standalone A–D token in the answer, so "I think it's B" → B. Otherwise, case-insensitive equality with an option's text. The citation comes from `query._context([hit])[1][0]`, where `hit` is built from the chunk and source title.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: generate quiz questions and grade answers (mcq by letter, short by rubric)`

---

### Task 5: Learner facts

**Files:** Create `core/memory.py`. Test `tests/test_memory.py`.

**Interfaces — Produces:** `should_remember(message: str) -> bool`; `facts() -> list[LearnerFact]`; `update(message: str) -> int`; `profile() -> str`; `delete(fact_id: int) -> bool`.

- [ ] **Step 1: Failing tests**
  - `test_should_remember`: `"My exam is on 12 Dec"`, `"I prefer short examples"` → True. `"what is an enzyme"`, `"B"` → False.
  - `test_update_applies_ops`: ADD 2 facts, then UPDATE id 1 and DELETE id 2 → one fact with the new text. Return values are 2 and 2.
  - `test_update_skips_invalid_ops`: an unknown id, an empty fact, op `"MERGE"`, and a non-JSON reply (the fake raises `ValueError`) → 0 applied, nothing changed.
  - `test_fact_cap`: with 20 facts, a further ADD is skipped.
  - `test_profile_text`: `""` when empty; `"About the student:\n- a\n- b"`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The prompt shows facts as `[id] fact` lines.
- [ ] **Step 4: Full suite.** Expected: all pass.
- [ ] **Step 5: Commit** `feat: learner facts with ADD/UPDATE/DELETE extraction`

---

### Task 6: Agent quiz sessions and memory

**Files:** Modify `core/agent.py`. Test `tests/test_agent.py`.

**Interfaces — Consumes:** `tutor.*` (Tasks 1, 3, 4) and `memory.*` (Task 5). **Produces:** state `quiz`; nodes `quiz_start`, `quiz_answer`, `quiz_stop`, `remember`; `LABELS` entries; `QUIZ_STOP` regex; `_quiz_about(message) -> Optional[str]`; constants for the spec strings. `deferred` and `DEFERRED` are removed. Update every reference, including the Phase 3 tests that assert `DEFERRED`.

- [ ] **Step 1: Failing tests.** Monkeypatch `tutor.pick_targets`, `tutor.generate_question` and `tutor.grade` with scripted fakes. Patch them as attributes of `core.tutor`, which `agent` calls through `tutor.<name>`.
  - `test_quiz_session_runs_to_summary`:
    - "quiz me on enzymes" → `about == "enzymes"` was passed; the answer is "Question 1/5 · …" with "A) … D)" and "Reply with a letter.";
    - next turns: "B" → "✅ Correct." + "Question 2/5"; …;
    - turn 5 → "Session done: 4/5 correct." and `quiz` None;
    - `tutor.record` was called 5 times (use the real `record`, then assert `QuizAttempt` rows).
  - `test_stop_ends_quiz` / `test_new_quiz_replaces_active_one`.
  - `test_failed_answer_turn_keeps_quiz`: `grade` raises → the run raises; the next turn's answer is graded against the same question.
  - `test_review_with_nothing_due_says_so`.
  - `test_remember_skips_quiz_answers_and_stores_facts` (Review Focus 3): "my answer is B" during a quiz → `memory.update` not called. "My exam is on 12 Dec" outside a quiz → called once. The remembered profile then appears in `generate`'s system prompt on the next qa turn.
  - `test_done_when_review_targets_planted_mistakes`, using the **real** tutor and memory with planted chunks and edges:
    - concepts A–E on 5 indexed chunks;
    - the fake LLM returns valid questions and rubric verdicts so that A is wrong, B partial and C–E correct, in a 5-question session started with "quiz me";
    - then "review" → the `quiz` targets are exactly `[A, B]` in that order, and the first question's concept is A.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** per the spec Agent section.
  - `route_intent` reads `state.get("quiz")` from the checkpoint and does not reset it.
  - `quiz_*` nodes return `answer`, `citations` and `quiz`.
  - `remember` checks `state["intent"]`.
  - `generate` appends `memory.profile()` to its system prompt.
  - The question text goes through one formatter, `_format_question(q, i, total)`.
- [ ] **Step 4: Full suite.** Expected: all pass. The Phase 3 tests asserting the old deferred reply are rewritten to assert the quiz start.
- [ ] **Step 5: Commit** `feat: agent quiz and review sessions; learner facts feed the tutor`

---

### Task 7: API, UI, README, real-model check

**Files:** Modify `api/main.py`, `ui/app.py` (sidebar), `README.md`. Test `tests/test_api.py`.

**Interfaces — Consumes:** `memory.facts`, `memory.delete`, `ConceptMastery`.

- [ ] **Step 1: Failing tests**
  - `test_learner_endpoint_orders_mastery_and_counts_due`;
  - `test_delete_fact` (200, then 404).
- [ ] **Step 2: Run.** Expected: FAIL (404).
- [ ] **Step 3: Implement.**
  - **Endpoints:** as in the spec.
  - **Sidebar expander "📈 Your progress":**
    - `api("GET", "/learner")` on each render;
    - "N concepts due today";
    - up to 5 mastery rows as `{emoji} concept — next review {date}`, with emoji ✅ / 🟡 / ❌ by `last_verdict`;
    - facts with a 🗑 `st.button(key=f"fact_{id}")` → DELETE, then `st.rerun()`.
  - **README:** the features row, Phase 5 usage ("quiz me on …", "review", "stop", the progress panel), and the learner-data note: attempts and mastery stay when sources are deleted.
- [ ] **Step 4: Offline checks.** `py_compile ui/app.py`, then the full suite. Expected: compiles; all pass.
- [ ] **Step 5: Real-model done-when** (by hand, on throwaway `DATABASE_URL` / `QDRANT_PATH` / `AGENT_DB_PATH` and an in-process script like Phase 4's `check.py`):
  - ingest two short notes with distinct concepts;
  - run `concepts.extract_pending()` so triples exist;
  - "quiz me" through 5 turns, answering two concepts wrong on purpose;
  - then "review".

  Pass: review's targets are exactly those two concepts. Record the questions, timings and pass/fail in the ledger.
- [ ] **Step 6: Commit** `feat: learner API and progress panel; README phase 5`
