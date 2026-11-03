# Phase 5 — Learner memory and adaptive tutoring: design

Date: 2026-11-02. Builds on Phase 4 (merged into main at 1a0fb7b).

## Goal

A second memory: what StudyMind knows about the learner, not about the material. Done when, after a
session with planted wrong answers, the next "review" session targets exactly those concepts.

## Decisions

| Question | Decision |
|---|---|
| Learner facts | An own `learner_fact` table, with an LLM proposing ADD / UPDATE / DELETE (Mem0's pattern). No Mem0 dependency. |
| Concept tags | Derived from the Phase 2 `concept_edge` triples: up to 3 per chunk, most frequent first. No new LLM calls. |
| Quiz UX | In the Agent chat: one question per turn, answered in the next message. |
| Session size | 5 questions (`SESSION_SIZE = 5`). |
| Question timing | Generated one at a time, as the session goes, not all up front. |
| Scheduler | SM-2. Quality is correct 5, partial 3, wrong 1. |

## Constraints

- Everything local; tests offline with the `tests/conftest.py` fakes.
- New tables only (`create_all()`); no new columns on existing tables. `ConceptMastery` and `LearnerFact`
  are keyed so that reused ids cause no harm: `QuizAttempt` and `LearnerFact` use
  `sqlite_autoincrement`, like `PendingFigure`.
- Dates are local calendar days (`date.today()`), stored as ISO `YYYY-MM-DD` strings.
- Write files as UTF-8.

## Storage (core/db.py)

```python
class QuizAttempt(SQLModel, table=True):      # sqlite_autoincrement
    id: Optional[int] = Field(default=None, primary_key=True)
    topic_id: Optional[str] = Field(default=None, index=True)
    concept: str = Field(index=True)            # normalised, as in concept_edge
    chunk_id: Optional[int] = None
    kind: str                                   # mcq | short
    question: str
    answer: str
    verdict: Optional[str] = None               # correct | partial | wrong | None (ungraded)
    feedback: str = ""
    created_at: str                             # ISO datetime

class ConceptMastery(SQLModel, table=True):
    concept: str = Field(primary_key=True)      # normalised
    ease: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    next_review: str                            # ISO date
    last_verdict: Optional[str] = None
    attempts: int = 0
    wrong: int = 0                              # wrong + partial count

class LearnerFact(SQLModel, table=True):      # sqlite_autoincrement
    id: Optional[int] = Field(default=None, primary_key=True)
    fact: str
    updated_at: str
```

Mastery is global per concept, across topics (a concept is the same knowledge wherever it appears).
Deleting a source does not delete attempts or mastery: they describe the learner, not the source.

## Concept tags (core/concepts.py)

`chunk_concepts(chunk_ids: list[int]) -> dict[int, list[str]]` returns, per chunk, up to 3 normalised
concepts. They are counted over the chunk's edges (`head_norm` and `tail_norm`), most frequent first,
with ties broken by first appearance. Chunks with no edges are absent from the result.

`chunks_for_concept(concept: str, topic_id: Optional[str]) -> list[int]` returns the chunk ids whose edges
mention the concept (head or tail), ordered by how many of their edges mention it.

## Tutor (core/tutor.py)

```python
class Question(BaseModel):
    kind: Literal["mcq", "short"]
    question: str
    options: list[str] = []          # 4 for mcq, shown as A–D
    answer: str                      # mcq: the correct option's letter; short: the expected answer
    difficulty: Literal["easy", "medium", "hard"]
    concept: str
    chunk_id: int
    citation: dict                   # query._context citation for the chunk
```

- **`pick_targets(topic_id, about=None, review=False, n=SESSION_SIZE) -> list[tuple[str, int]]`**
  - **Quiz on a subject ("quiz me on X"):** retrieve `about` with
    `get_retriever(rewrite=False).retrieve(about, topic_id, k=12)`, keep `relevant()` hits, and take
    `chunk_concepts` of those chunks in rank order. The result is distinct concepts with the chunk they
    came from, at most `n`.
  - **Quiz with no subject ("quiz me"):** passages rarely contain their topic's name, so there is no search.
    Take the scope's concepts from `concept_edge` (the topic's, or all of them). Order them: never quizzed
    first (most mentioned first, then first seen), then the quizzed ones by lowest ease. Each comes with the
    chunk that mentions it most.
  - **Review:** `ConceptMastery` rows where `next_review <= today` or `last_verdict in ("wrong", "partial")`,
    ordered by `(next_review, ease, -wrong, concept)`. When `topic_id` is set, only concepts with a chunk in that
    topic count. Each concept gets its best chunk from `chunks_for_concept`. At most `n`. An empty list
    means nothing to review.
- **`generate_question(concept, chunk_id, kind, facts) -> Optional[Question]`** makes one `complete_json`
  call over the chunk text, asking for the JSON fields above.
  - The result is valid only when `kind` matches the request, an mcq has 4 options and an answer letter
    A–D, and every text field is non-empty.
  - One retry; then `None`.
  - `kind` alternates `mcq`, `short`, `mcq`, … by question number.
- **`grade(question, answer, facts) -> tuple[Optional[str], str]`** returns `(verdict, feedback)`.
  - **Mcq:** the chosen letter is found in this order:
    1. a reply that is just one letter A–D (any case, after stripping `()` and `.`) is that letter;
    2. otherwise, a reply that starts with a letter and `)`, `.` or `:` ("d) zinc", "(c) lead") is that letter;
    3. otherwise, a case-insensitive match with an option's text ("Vitamin C" is that option, not letter C);
    4. otherwise, the first standalone uppercase A–D in the reply ("I think it's B because a cell…" → B).

    The answer is correct when that letter is the right one.
    - Correct: `("correct", "")` with no LLM call.
    - Wrong: one `complete` call explains why the right option is correct, from the chunk, in at most
      2 sentences. The verdict is `"wrong"`.
  - **Short:** one `complete_json` rubric call with the chunk, the expected answer and the learner's
    answer → `{"verdict": "correct|partial|wrong", "feedback": "one line"}`. An invalid reply after
    one retry gives `(None, "Couldn't grade this answer.")`.
- **`record(question, answer, verdict, feedback, topic_id)`** inserts a `QuizAttempt`. When `verdict`
  is not None, it updates `ConceptMastery` with SM-2 (`sm2` below), creating the row on first sight.
- **`sm2(row, verdict, today) -> row`**, with `q = {"correct": 5, "partial": 3, "wrong": 1}[verdict]`:
  - When `q < 3`: `repetitions = 0`, `interval_days = 1`.
  - Otherwise: `repetitions += 1`, and `interval_days` is 1 on the first repetition, 6 on the second,
    else `round(interval_days * ease)`, using the ease from before this answer.
  - `ease = max(1.3, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))`.
  - `next_review = today + interval_days`; `last_verdict = verdict`; `attempts += 1`;
    `wrong += verdict != "correct"`.

## Learner facts (core/memory.py)

- **`should_remember(message) -> bool`**: true when the message has a first-person cue,
  `\b(i|i'm|im|my|me|i've|i'll)\b` or `\bexam\b`, case-insensitive. The agent never calls it for quiz
  answers.
- **`facts() -> list[LearnerFact]`**: all facts, oldest first.
- **`update(message) -> int`** makes one `complete_json` call with the numbered current facts and the
  message, and returns how many operations were applied.
  - The reply is `{"ops": [{"op": "ADD", "fact": "..."}, {"op": "UPDATE", "id": 3, "fact": "..."},
    {"op": "DELETE", "id": 2}]}`.
  - Invalid operations (an unknown id, an empty fact, an unknown op) are skipped. An invalid reply
    applies nothing.
  - At most `MAX_FACTS = 20`: an ADD beyond that is skipped.
- **`profile() -> str`**: `"About the student:\n- fact\n- fact"`, or `""` when there are none.

## Agent (core/agent.py)

- **State** gains `quiz: Optional[dict]`: `{"mode": "quiz"|"review", "topic_id", "targets": [[concept, chunk_id]],
  "index": int, "current": Question dict, "results": [verdict or None]}`. `route_intent` does not reset it.
- **Routing:**
  - If `state.get("quiz")` is set and the whole message is a stop command,
    `^\s*(stop|end( the)? quiz|quit)\s*[.!]?\s*$` → intent `quiz_stop`.
    "Stop codons end translation" is an answer.
  - Else, if a quiz is active and the whole message is not a new-session command
    (`review`, `review my weak topics`, `quiz me`, `quiz me on …`) → intent `quiz_answer`.
  - Else: `classify`, as now. A `quiz` or `review` intent → `quiz_start` (it replaces any active quiz).
- **Quiz topic:** `about` is the text after `on` / `about` in the message (for example "quiz me on enzymes"
  → "enzymes"), else None.
- **Nodes** (they replace `deferred` and `DEFERRED`):
  - **`quiz_start`:**
    - picks targets;
    - walks the targets until `generate_question` succeeds (failed targets are dropped);
    - replies with the question text; `citations = [question.citation]`; sets `quiz`.
    - No targets → the answer is `"Nothing to review yet — answer some quiz questions first, or try 'quiz me on <topic>'."`
      for review, or `"I couldn't find quizzable material for that yet."` for a quiz.
  - **`quiz_answer`:**
    - grades and records the answer;
    - replies `✅ Correct.` / `🟡 Partly right. <feedback>` / `❌ Not quite — the answer is <answer>. <feedback>`
      (or `⚠️ Couldn't grade this answer.`);
    - then shows the next question as in `quiz_start`, or, when the targets are used up, a summary:
      `"Session done: 3/5 correct. To review: concept (tomorrow), …"`. `quiz` is then set to None.
  - **`quiz_stop`:** replies `"Quiz stopped after N questions."` and sets `quiz` to None.
  - **`remember`:** runs after `respond`. When the intent is not `quiz_answer` and
    `memory.should_remember(question)`, it calls `memory.update(question)`. Errors are logged, never raised.
- **Question format:**
  - `"Question {i}/{total} · {concept} · {difficulty}\n\n{question}"`, plus for mcq
    `"\n\nA) …\nB) …\nC) …\nD) …\n\nReply with a letter."`.
  - Short questions end with `"\n\nAnswer in a sentence or two."`.
- **Facts in prompts:** `generate` (answers and explanations) appends `memory.profile()` to its system
  prompt when it is non-empty. `generate_question` and `grade` take `facts` the same way.
- **Edges:**
  - `route_intent → quiz_start | quiz_answer | quiz_stop | retrieve | connect`;
  - every quiz node → `respond`;
  - `respond → remember → END`.
- **Labels:** `quiz_start` "making a question", `quiz_answer` "grading", `quiz_stop` "stopping quiz",
  `remember` "remembering".

## API

- `GET /learner` → `{"facts": [{"id", "fact", "updated_at"}], "mastery": [...ConceptMastery fields],
  "to_review": int}`. Mastery is ordered as review would pick it (`next_review, ease, -wrong, concept`) and
  leaves out concepts with no chunk left (their source was deleted). `to_review` counts with the review rule:
  `next_review <= today` or the last verdict was wrong or partial.
- `DELETE /learner/facts/{id}` → 404 for an unknown id.

## UI

The sidebar gets a "📈 Your progress" expander:
- "N concepts to review" (the review rule, as `to_review`);
- the first 5 mastery rows (concept, last result emoji, next review);
- the facts, each with a 🗑 button that calls the DELETE endpoint.

Quiz questions and feedback are ordinary Agent chat messages. No other UI changes.

## Errors

- Question generation fails for every target → the message says so, and no quiz is started.
- Grading fails → an attempt is recorded with `verdict=None` and no mastery change, and the session moves on.
- An LLM exception inside a node → the existing `/agent` error event. Because the checkpoint is not
  advanced, `quiz` stays as it was and the learner can answer again.
- Memory update failures are logged only.

## Testing (offline)

1. `sm2`: correct ×3 → intervals 1, 6, 16 and ease 2.6, 2.7, 2.8 (the third interval is round(6 × 2.7)).
   Partial → interval 1, ease drops by 0.14. Wrong → interval 1, ease drops by 0.54 (floor 1.3).
2. `chunk_concepts` / `chunks_for_concept`: top 3 by frequency; chunks without edges are absent.
3. `pick_targets(review=True)`: only due or weak concepts, in `(next_review, ease, -wrong, concept)` order;
   `[]` when there are none; respects `topic_id`.
4. `grade`:
   - mcq by letter (`"b"`, `"(B)"`, `"B."`) and by option text; no LLM call when correct;
   - a wrong mcq makes one explain call;
   - a short rubric gives each verdict;
   - an invalid rubric twice → `(None, ...)`.
5. `generate_question`: valid JSON → a Question; an invalid mcq (3 options) twice → None.
6. Memory: ADD / UPDATE / DELETE applied; invalid ops skipped; the cap of 20; `should_remember` cases;
   `profile()` text reaches `generate`'s system prompt.
7. Agent flow:
   - "quiz me on enzymes" → Q1 (mcq) → "B" → feedback + Q2 (short) → … → summary, with `quiz` None;
   - "stop" mid-quiz → stopped;
   - a new "quiz me" mid-quiz replaces the session;
   - an LLM error during `quiz_answer` keeps the quiz state.
8. **Done-when simulation:** a 5-question session over concepts A–E. The answer to A is wrong, to B
   partly right, and C–E are correct. Then "review" → targets exactly `[A, B]`. A comes first because its
   ease is lower (1.96 against 2.36); C–E are not due until tomorrow and their last verdict was correct.
9. API: `GET /learner` ordering and `due`; `DELETE /learner/facts/{id}` 200 / 404.

Done-when, real model, checked by hand: a throwaway DB with notes on two topics; quiz yourself, answer two
concepts wrong on purpose, then "review" and confirm the agent asks about exactly those.

## Out of scope

- Knowledge-tracing models beyond SM-2.
- Mem0 itself.
- Review reminders or notifications.
- Adaptive difficulty.
- Editing facts in the UI (delete only).
- Per-topic mastery.
