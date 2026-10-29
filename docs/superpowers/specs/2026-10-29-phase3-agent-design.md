# Phase 3 — The LangGraph agent: design

Date: 2026-10-29. Builds on Phase 2 (merged at 4df1d30).

## Goal

One agent decides what to do with a message, instead of a fixed pipeline per tab.
Done when:

1. a vague question triggers a visible rewrite-and-retry in the trace, and
2. the self-check catches a planted unsupported claim in a unit test.

## Decisions

| Question | Decision |
|---|---|
| Dependencies | Add `langgraph` and `langgraph-checkpoint-sqlite` (pinned). |
| Phase 5 tools (`generate_quiz`, `grade_answer`, `get_weak_concepts`, `schedule_review`) | Deferred. The quiz and review intents reply "coming in Phase 5". `learner_profile` is not in the state yet. |
| grade_documents | Reranker-score cutoff, with no LLM call. The same cutoff fixes `connect_topics`. |
| self_check | One batched `complete_json` call per answer. |
| route_intent | Rules plus an optional UI hint, with no LLM call. |
| UI | Add an Agent chat tab and keep Ask and Connect. |
| Graph style | An explicit `StateGraph`. Tools are plain async functions called by nodes, not LLM tool-calling: Qwen 7B tool-calling is unreliable, and rules route already. |

## Constraints

- Everything local: Qwen 2.5 7B via Ollama, fastembed on CPU.
- Tests stay offline with the existing fakes in `tests/conftest.py`.
- The checkpointer gets its own SQLite file, `data/agent.db`, so it does not contend with `studymind.db`.
- Write files as UTF-8 explicitly.

## Setting

`relevance_min: float = -3.0` in `core/config.py` (env `RELEVANCE_MIN`). It sits between the measured
top reranker logits: related -1.6 to 4.0, unrelated -5.2 to -6. Hits are relevant when
`score >= relevance_min`. The cutoff applies to hybrid only: dense scores are cosine, so with
`RETRIEVER=dense` every hit counts as relevant.

`core/retrieval.py` gets one helper used by both the agent and `connect_topics`:

```python
def relevant(hits: list[Hit]) -> list[Hit]:
    """Hits that clear the reranker cutoff; all hits for the dense retriever (cosine scores)."""
```

## Fix carried in: connect_topics

`connect_topics` filters `hits_a` and `hits_b` through `relevant()` before computing the overlap.
An unrelated topic then retrieves no chunks, finds no shared concepts, and gets the existing
"No connections found" summary.

## core/agent.py

### State

```python
class StudyState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # {"role", "content"}; appended per turn
    topic_id: Optional[str]
    topics: list[str]          # two topic names for connect, from the UI
    intent: str                # qa | explain | connect | quiz | review
    question: str              # this turn's user message
    query: str                 # current search query (question, or a rewrite)
    queries_tried: list[str]
    documents: list[dict]      # hits as dicts, so the checkpointer can serialise them
    answer: str
    citations: list[dict]
    retries: int
    regenerated: bool
    unsupported: list[str]
    warning: Optional[str]
```

Fields for one turn (`query`, `queries_tried`, `documents`, `retries`, `regenerated`, `unsupported`,
`warning`, `answer`, `citations`) are reset by `route_intent` at the start of each turn. `messages`
accumulates across turns in a thread.

### Tool

`search_material(query, topic_id) -> list[dict]` calls `get_retriever().retrieve(query, topic_id)`
and returns the hits as dicts (`dataclasses.asdict`). Phase 5 adds its tools beside it.

### Nodes

- **route_intent** (no LLM). The UI `intent` hint wins when one is given. Otherwise the first
  match wins, case-insensitive:
  - `quiz me|test me|quiz` → quiz
  - `\breview\b` → review
  - two `topics` given, or `between (.+) and (.+)` / `connect (.+) (with|and|to) (.+)` → connect (the parsed names fill `topics`)
  - `^(explain|why|how does|how do|what is|what are)\b` → explain
  - otherwise → qa

  It also appends the user message, sets `question` and `query` to the message, and resets the per-turn fields.
- **retrieve**: `documents = await search_material(query, topic_id)`, and appends `query` to `queries_tried`.
- **grade_documents**: keeps only `relevant()` documents. The edge function reads the result.
- **rewrite_query** (one LLM call). The prompt holds the last 4 messages, the question and `queries_tried`,
  and asks for one short, specific search query. If the reply is empty after stripping, or repeats a
  query already tried, the question is used. Increments `retries`.
- **generate** (one LLM call). `_context(hits)` and `ANSWER_SYSTEM` come from `core.query`. The explain
  intent uses `EXPLAIN_SYSTEM = CITE_RULES + "Explain step by step for a student, ..." + the same
  not-in-material sentence`. When `unsupported` is set (a regeneration), the prompt adds:
  "These sentences were not supported by the passages; drop or fix them: ...". Appends the
  assistant message.
- **self_check** (one `complete_json` call). Input: the answer and the numbered passages. Output:
  `{"unsupported": ["sentence", ...]}`. It is skipped (supported) when the answer starts with
  "This isn't in your material". A parse failure counts as supported and is logged.
- **not_found**: answer = "This isn't in your material." plus the queries tried. No LLM call.
- **connect**: calls `query.connect_topics(a, b)`. Answer = its `summary`, plus `graph` = `html_graph`
  for the UI.
- **deferred**: answer = "Quizzes and reviews are coming in Phase 5."

### Edges

```
START → route_intent
route_intent → retrieve | connect | deferred          (by intent; qa and explain → retrieve)
retrieve → grade_documents
grade_documents → generate          if any relevant documents
                → rewrite_query     if none and retries < 2
                → not_found         if none and retries == 2
rewrite_query → retrieve
generate → self_check
self_check → END                    if unsupported is empty
           → generate               if unsupported and not regenerated (sets regenerated)
           → END with warning       if unsupported and regenerated
connect, deferred, not_found → END
```

The warning text is "Some statements may not be supported by your material: ...", listing the flagged
sentences. The loops are bounded by `retries` and `regenerated`, not by `recursion_limit`.

### Graph construction

`build(checkpointer) -> CompiledGraph`. `open_graph()` is an async context manager that opens
`AsyncSqliteSaver.from_conn_string(DATA_DIR / "agent.db")` and yields the compiled graph. The path
comes from a new setting, `agent_db_path`, which tests point at `tmp_path`.

Node labels for the trace:
`{"route_intent": "routing", "retrieve": "searching", "grade_documents": "checking sources",
"rewrite_query": "rewriting query", "generate": "answering", "self_check": "verifying",
"not_found": "nothing found", "connect": "connecting topics", "deferred": "not yet available"}`.

### Latency on Qwen 7B (expected)

- Typical qa or explain turn: generate plus self_check, about 30 s.
- Worst case: 2 rewrites plus a regeneration, about 90 s or more.
- Connect, deferred and not_found with no rewrite: no LLM call.

## API

- `POST /agent`, body `{message, thread_id, topic_id?, intent?, topics?}`. The response is a
  `StreamingResponse` with media type `application/x-ndjson`, produced by
  `graph.astream(input, config={"configurable": {"thread_id": ...}}, stream_mode="updates")`.
  - One line per node: `{"event": "node", "node": name, "label": label, "query": ...}`. `query` is
    present for retrieve and rewrite_query, which is what makes a retry visible.
  - Final line: `{"event": "done", "intent", "answer", "citations", "warning", "graph"}`.
  - An exception inside the graph yields `{"event": "error", "error": "Type: msg"}` and ends the stream.
- `GET /agent/{thread_id}` returns `{"messages": [...]}` from the checkpoint state, or `[]`.
- `lifespan` enters `agent.open_graph()` and stores the graph on `app.state.agent`.

## UI

A new tab, "🤖 Agent", placed second.

- It keeps `agent_thread_id` (a uuid4) in `session_state`. On first render it loads its history
  from `GET /agent/{id}`. A "New conversation" button issues a new id.
- It renders the history with `st.chat_message`, and takes input from `st.chat_input` (`key="agent_input"`).
- On send, `httpx.stream("POST", "/agent", timeout=900)` iterates over the lines. Each node event updates an
  `st.status` box. Rewrites show the new query. On `done` it renders the answer, any warning as
  `st.warning`, citation chips via `_render_citations`, and the graph HTML for connect.
- `topic_id` is the sidebar topic (`None` for all topics). An optional "Connect with…" multiselect
  of two topics sends `topics`.

## Testing (offline; tests/test_agent.py)

Fakes: the existing conftest fixtures. Tests set fake rerank scores and fake `llm.complete` /
`complete_json` replies per case. The checkpointer is `AsyncSqliteSaver` on `tmp_path`.

1. Routing: one case per intent, plus the UI hint overriding the rules.
2. Corrective loop: a weak, then strong score leads to retrieve → rewrite → retrieve → generate. Always
   weak leads to 2 rewrites and then the not-found answer listing the queries tried.
3. Self-check (done-when 2): a planted unsupported sentence is flagged. The graph regenerates once with the
   feedback, and a second flag ends with `warning` set. A clean answer ends with no warning.
4. self_check is skipped for "This isn't in your material."
5. Persistence: two turns on one `thread_id`; the second sees both prior messages.
6. `relevant()`: the cutoff applies under hybrid and is a pass-through under dense.
7. `connect_topics` with below-cutoff hits returns "No connections found".
8. API: `/agent` streams node events and then `done`. A node raising produces an `error` event.
   `GET /agent/{id}` returns the history.

Done-when 1 is checked by hand against the real model with a throwaway `DATABASE_URL`, `QDRANT_PATH`
and `AGENT_DB_PATH`.

## Out of scope

- The Phase 5 tools and `learner_profile`.
- An LLM intent classifier and LLM document grading.
- Removing the Ask and Connect tabs.
- Recalibrating `relevance_min` beyond the measured bands. Re-check it once the eval set grows.
