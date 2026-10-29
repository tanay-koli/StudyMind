# Phase 3 — LangGraph Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One LangGraph agent routes each chat message (qa / explain / connect / quiz / review). It does corrective retrieval (rewrite and retry), checks its answer against the cited chunks, and streams its steps to a new Agent tab.

**Architecture:** An explicit `StateGraph` in `core/agent.py`. Its nodes are plain async functions over the existing `core.retrieval` / `core.query` / `core.llm` code. Routing is by rules and document grading is by reranker cutoff, so no LLM call is spent on either. `AsyncSqliteSaver` persists threads in `data/agent.db`. FastAPI streams node events as NDJSON, and Streamlit renders them in an `st.status` box.

**Tech Stack:** Python 3.13, langgraph 1.2.18, langgraph-checkpoint-sqlite 3.1.2, FastAPI, Streamlit, LiteLLM → Ollama qwen2.5:7b, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-29-phase3-agent-design.md`

## Global Constraints

- Everything local: Qwen 2.5 7B via Ollama; tests are offline with the fakes in `tests/conftest.py` (no network, no model downloads).
- Async tests use `@pytest.mark.asyncio` explicitly (there is no auto mode).
- Fake `llm.complete` replacements must accept `system` positionally: `async def f(prompt, system=None, **kw)`.
- Write files as UTF-8 explicitly. The repo has CRLF files, so don't use shell heredocs with `\n` escapes for code.
- Run tests with `.venv/Scripts/python -m pytest -q`. The baseline is 130 passing, and every task ends green.
- `relevance_min: float = -3.0`. The cutoff applies to hybrid only; under dense every hit is relevant.
- Retries: at most 2 rewrites. Regenerations: at most 1.
- Exact copy: not-found answer starts with `This isn't in your material.`; the deferred answer is `Quizzes and reviews are coming in Phase 5.`; the warning starts with `Some statements may not be supported by your material:`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **No material at all** (empty index, or a topic with no chunks): retrieve returns zero hits. Rewriting cannot help, so go straight to not_found with no LLM call. This deliberately refines the spec: only *low-scoring* hits trigger a rewrite. Test in Task 3.
2. **Ollama down or an LLM exception mid-turn**: the stream emits an `error` event and closes, and the next turn on the same thread still works. Tests in Task 4.
3. **self_check returns odd JSON** (`{"unsupported": "none"}`, a missing key, non-string items): anything that isn't a list of strings counts as `[]` (supported). Test in Task 3.
4. **Connect with one topic** (the UI sends 1 topic, or the regex captures only one): route as qa, never call `connect_topics` with an empty name. Test in Task 2.
5. **Follow-up turn on a thread**: the second turn's per-turn fields (`retries`, `regenerated`, `warning`, `unsupported`, `queries_tried`) start fresh. A warning from turn 1 must not leak into turn 2. Test in Task 3.

---

### Task 1: Dependencies, `relevant()` cutoff, connect_topics fix

**Files:**
- Modify: `requirements.txt` (add after the LiteLLM block, with a comment `# Agent (Phase 3)`)
- Modify: `core/config.py` (`Settings`: retrieval block)
- Modify: `core/retrieval.py` (after `Hit` / `Retriever`)
- Modify: `core/query.py` (`connect_topics`)
- Test: `tests/test_retrieval.py`, `tests/test_query.py`

**Interfaces:**
- Produces: `settings.relevance_min: float = -3.0`; `settings.agent_db_path: str = (DATA_DIR / "agent.db").as_posix()`; `retrieval.relevant(hits: list[Hit]) -> list[Hit]` (also accepts hit dicts: compare `h.score if isinstance(h, Hit) else h["score"]`).

- [ ] **Step 1: Install and pin the dependencies**

Add `langgraph==1.2.18` and `langgraph-checkpoint-sqlite==3.1.2` to `requirements.txt`, then run
`.venv/Scripts/pip install langgraph==1.2.18 langgraph-checkpoint-sqlite==3.1.2`.
Verify: `.venv/Scripts/python -c "from langgraph.graph import StateGraph; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; print('ok')"` prints `ok`.
If the install fails on Python 3.13 / Windows, stop and report. Don't downgrade without asking.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_retrieval.py
from core.retrieval import Hit, relevant
from core.config import settings

def _h(score): return Hit(1, 1, "T", 1, None, "x", score)

def test_relevant_cuts_below_threshold_for_hybrid(monkeypatch):
    monkeypatch.setattr(settings, "retriever", "hybrid")
    assert [h.score for h in relevant([_h(2.0), _h(-3.0), _h(-5.5)])] == [2.0, -3.0]

def test_relevant_passes_everything_for_dense(monkeypatch):
    monkeypatch.setattr(settings, "retriever", "dense")
    assert len(relevant([_h(0.1), _h(-9.0)])) == 2

def test_relevant_accepts_dicts(monkeypatch):
    monkeypatch.setattr(settings, "retriever", "hybrid")
    assert relevant([{"score": -6.0}, {"score": 0.0}]) == [{"score": 0.0}]
```

```python
# tests/test_query.py — beside test_connect_no_overlap_empty_state
@pytest.mark.asyncio
async def test_connect_ignores_irrelevant_hits(fake, monkeypatch):
    # Same planted overlap as test_connect_builds_graph_without_llm, but topic b's hits score below the cutoff
    monkeypatch.setattr(query.settings, "retriever", "hybrid")
    ids = _planted(4, {0: [("Gradient", "drives", "Learning")], 2: [("Optimiser", "follows", "gradient")]})
    _two_topics(fake, ids)  # then set the score of every ("b", None) hit to -6.0 on the fake retriever
    res = await query.connect_topics("a", "b")
    assert res["connection_method"] == "none" and res["nodes"] == []
    assert "No connections found between 'a' and 'b'" in res["summary"]
```

(The `_two_topics` / `fake` helpers already exist in `tests/test_query.py`. Read them, then lower the b-hit scores through whatever the helper returns.)

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_retrieval.py tests/test_query.py -k "relevant or irrelevant"`
Expected: an ImportError on `relevant`, and the connect test fails (`concept_overlap`).

- [ ] **Step 4: Implement the settings, `relevant()`, and filter `hits_a` / `hits_b` through `relevant()` in `connect_topics`, right after the `gather`**

The filter goes before the `pending_count` ids are built, so the pending note only counts relevant chunks.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (130 + 4). The existing connect tests use score 1.0 and the fake reranker scores are ≥ 0, so they still pass.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt core/config.py core/retrieval.py core/query.py tests/test_retrieval.py tests/test_query.py
git commit -m "feat: reranker relevance cutoff; connect_topics ignores irrelevant hits"
```

---

### Task 2: Agent state and intent routing rules

**Files:**
- Create: `core/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces:
  - `StudyState(TypedDict, total=False)`, with fields exactly as in the spec's State section, plus `graph: Optional[str]` (connect HTML) and `intent_hint: Optional[str]` (the UI hint, input only).
  - `classify(message: str, topics: list[str] | None, hint: str | None) -> tuple[str, list[str]]`. It returns `(intent, topics)`, where `intent ∈ {"qa","explain","connect","quiz","review"}` and topics are the two connect names (otherwise `[]`).
  - `LABELS: dict[str, str]`: the node → trace-label map, verbatim from the spec, plus `"respond": "done"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent.py
import pytest
from core import agent

@pytest.mark.parametrize("msg,topics,hint,expected", [
    ("quiz me on attention", None, None, ("quiz", [])),
    ("Can you test me?", None, None, ("quiz", [])),
    ("I want to review today", None, None, ("review", [])),
    ("what links these?", ["Biology", "Chemistry"], None, ("connect", ["Biology", "Chemistry"])),
    ("What is the connection between attention and memory?", None, None, ("connect", ["attention", "memory"])),
    ("connect enzymes with catalysts", None, None, ("connect", ["enzymes", "catalysts"])),
    ("Explain backprop", None, None, ("explain", [])),
    ("how does dropout work", None, None, ("explain", [])),
    ("list the layers in the encoder", None, None, ("qa", [])),
    ("Explain backprop", None, "qa", ("qa", [])),            # UI hint wins
    ("what links these?", ["Biology"], None, ("qa", [])),     # Review Focus 4: one topic is not connect
])
def test_classify(msg, topics, hint, expected):
    assert agent.classify(msg, topics, hint) == expected
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/Scripts/python -m pytest -q tests/test_agent.py`
Expected: ImportError / AttributeError on `classify`.

- [ ] **Step 3: Implement `StudyState`, `LABELS`, `classify` in `core/agent.py`**

The rule order and regexes come from the spec's route_intent section, matched case-insensitively. Strip `?.!` and whitespace from the captured connect names. A hint of `connect` with fewer than 2 topics also falls back to the rules.
The module docstring follows the house style ("StudyMind — Agent", one paragraph).

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest -q tests/test_agent.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/agent.py tests/test_agent.py
git commit -m "feat: agent state and rule-based intent routing"
```

---

### Task 3: Graph nodes, edges, checkpointer

**Files:**
- Modify: `core/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `classify`, `LABELS`, `StudyState` (Task 2); `retrieval.relevant` (Task 1); `query._context`, `query.ANSWER_SYSTEM`, `query.CITE_RULES`, `query.connect_topics`; `llm.complete`, `llm.complete_json`; `settings.agent_db_path`.
- Produces:
  - `async def search_material(query: str, topic_id: str | None) -> list[dict]`, the tool (hits as `dataclasses.asdict`).
  - `def build(checkpointer) -> CompiledStateGraph`.
  - `@asynccontextmanager async def open_graph(path: str | None = None)`, which yields the compiled graph backed by `AsyncSqliteSaver.from_conn_string(path or settings.agent_db_path)`.
  - The graph input per turn: `{"messages": [], "question": msg, "topic_id": tid, "topics": topics or [], "intent_hint": hint}`. `route_intent` appends `{"role": "user", "content": question}` itself.
  - Final state keys read by the API: `intent, answer, citations, warning, graph, messages`.
  - Node names exactly: `route_intent, retrieve, grade_documents, rewrite_query, generate, self_check, not_found, connect, deferred, respond`.
  - `NOT_FOUND = "This isn't in your material."`, `DEFERRED = "Quizzes and reviews are coming in Phase 5."`, `WARNING = "Some statements may not be supported by your material:"`.

- [ ] **Step 1: Write the failing tests**

Shared helpers in `tests/test_agent.py`:

```python
import itertools
from core.llm import llm

def _doc(score, text="Attention weighs tokens.", cid=1):
    return {"chunk_id": cid, "source_id": 1, "source_title": "Notes", "page": 2, "section": None,
            "text": text, "score": score}

@pytest.fixture
def script(monkeypatch, tmp_path):
    """Scripted search results and LLM replies; returns a runner and the call log."""
    log = {"queries": [], "prompts": [], "checks": []}
    s = {"search": [], "complete": [], "json": []}

    async def search(q, topic_id):
        log["queries"].append(q)
        return s["search"].pop(0) if s["search"] else []

    async def complete(prompt, system=None, **kw):
        log["prompts"].append((prompt, system))
        return s["complete"].pop(0)

    async def complete_json(prompt, system=None, **kw):
        log["checks"].append(prompt)
        return s["json"].pop(0) if s["json"] else {"unsupported": []}

    monkeypatch.setattr(agent, "search_material", search)
    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "complete_json", complete_json)
    monkeypatch.setattr(agent.settings, "retriever", "hybrid")

    async def run(msg, thread="t1", **kw):
        async with agent.open_graph(str(tmp_path / "agent.db")) as g:
            cfg = {"configurable": {"thread_id": thread}}
            nodes = [n async for ev in g.astream({"messages": [], "question": msg, "topics": [], **kw}, cfg,
                                                  stream_mode="updates") for n in ev]
            return nodes, (await g.aget_state(cfg)).values
    return s, log, run
```

Tests (assertions are exact):

```python
@pytest.mark.asyncio
async def test_strong_retrieval_answers_without_rewrite(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]; s["complete"] = ["Attention weighs tokens [1]."]
    nodes, st = await run("what does attention do")
    assert nodes == ["route_intent", "retrieve", "grade_documents", "generate", "self_check", "respond"]
    assert st["answer"] == "Attention weighs tokens [1]." and st["warning"] is None
    assert st["citations"][0]["chunk_id"] == 1
    assert st["messages"][-1] == {"role": "assistant", "content": "Attention weighs tokens [1]."}

@pytest.mark.asyncio
async def test_weak_then_strong_rewrites_once(script):          # done-when 1, offline
    s, log, run = script
    s["search"] = [[_doc(-6.0)], [_doc(1.5)]]
    s["complete"] = ["self-attention mechanism transformer", "Answer [1]."]
    nodes, st = await run("that thing from lecture")
    assert nodes.count("retrieve") == 2 and nodes.count("rewrite_query") == 1
    assert log["queries"] == ["that thing from lecture", "self-attention mechanism transformer"]

@pytest.mark.asyncio
async def test_always_weak_gives_up_after_two_rewrites(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)]] * 3; s["complete"] = ["q2", "q3"]
    nodes, st = await run("vague")
    assert nodes.count("rewrite_query") == 2 and "not_found" in nodes and "generate" not in nodes
    assert st["answer"].startswith("This isn't in your material.")
    assert all(q in st["answer"] for q in ["vague", "q2", "q3"])

@pytest.mark.asyncio
async def test_zero_hits_skips_rewrite(script):                 # Review Focus 1
    s, log, run = script
    nodes, st = await run("anything")
    assert "rewrite_query" not in nodes and log["prompts"] == []
    assert st["answer"].startswith("This isn't in your material.")

@pytest.mark.asyncio
async def test_empty_rewrite_falls_back_to_question(script):
    s, log, run = script
    s["search"] = [[_doc(-6.0)], [_doc(2.0)]]; s["complete"] = ["   ", "A [1]."]
    await run("vague")
    assert log["queries"] == ["vague", "vague"]

@pytest.mark.asyncio
async def test_self_check_catches_planted_claim(script):         # done-when 2
    s, log, run = script
    planted = "Attention was invented in 1850 [1]."
    s["search"] = [[_doc(2.0)]]
    s["complete"] = [f"Attention weighs tokens [1]. {planted}", f"Attention weighs tokens [1]. {planted}"]
    s["json"] = [{"unsupported": [planted]}, {"unsupported": [planted]}]
    nodes, st = await run("what does attention do")
    assert nodes.count("generate") == 2 and nodes.count("self_check") == 2
    assert planted in log["prompts"][1][0]                       # the feedback reached the regeneration
    assert st["warning"].startswith("Some statements may not be supported by your material:") and planted in st["warning"]
    assert [m["role"] for m in st["messages"]] == ["user", "assistant"]   # one reply despite regeneration

@pytest.mark.asyncio
async def test_regeneration_that_fixes_the_claim_clears_warning(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]; s["complete"] = ["Bad [1].", "Good [1]."]
    s["json"] = [{"unsupported": ["Bad [1]."]}, {"unsupported": []}]
    _, st = await run("q")
    assert st["answer"] == "Good [1]." and st["warning"] is None

@pytest.mark.asyncio
async def test_self_check_skipped_for_not_in_material(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]; s["complete"] = ["This isn't in your material."]
    nodes, _ = await run("q")
    assert "self_check" in nodes and log["checks"] == []        # node runs, makes no call

@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [{"unsupported": "none"}, {}, {"unsupported": [3, None]}])
async def test_self_check_odd_json_counts_as_supported(script, reply):   # Review Focus 3
    s, log, run = script
    s["search"] = [[_doc(2.0)]]; s["complete"] = ["A [1]."]; s["json"] = [reply]
    nodes, st = await run("q")
    assert nodes.count("generate") == 1 and st["warning"] is None

@pytest.mark.asyncio
async def test_explain_uses_explain_prompt(script):
    s, log, run = script
    s["search"] = [[_doc(2.0)]]; s["complete"] = ["Step 1 [1]."]
    await run("explain attention")
    assert log["prompts"][0][1] == agent.EXPLAIN_SYSTEM

@pytest.mark.asyncio
async def test_deferred_and_connect_make_no_llm_call(script, monkeypatch):
    s, log, run = script
    async def fake_connect(a, b): return {"summary": f"{a}~{b}", "html_graph": "<g/>"}
    monkeypatch.setattr(agent.query, "connect_topics", fake_connect)
    _, st = await run("quiz me", thread="q")
    assert st["answer"] == "Quizzes and reviews are coming in Phase 5."
    _, st = await run("connect enzymes with catalysts", thread="c")
    assert st["answer"] == "enzymes~catalysts" and st["graph"] == "<g/>" and log["prompts"] == []

@pytest.mark.asyncio
async def test_second_turn_sees_history_and_fresh_turn_fields(script):   # spec test 5 + Review Focus 5
    s, log, run = script
    s["search"] = [[_doc(2.0)], [_doc(-6.0)], [_doc(2.0)]]
    s["complete"] = ["Bad [1].", "Bad [1].", "its complexity rewrite", "Fine [1]."]
    s["json"] = [{"unsupported": ["Bad [1]."]}, {"unsupported": ["Bad [1]."]}, {"unsupported": []}]
    _, st1 = await run("what is attention")
    assert st1["warning"]
    _, st2 = await run("what about its complexity?")
    assert len(st2["messages"]) == 4 and st2["warning"] is None and st2["retries"] == 1
    assert "what is attention" in log["prompts"][2][0]         # rewrite prompt carries prior messages
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_agent.py`
Expected: the new tests fail (`open_graph` missing); the Task 2 tests still pass.

- [ ] **Step 3: Implement the nodes, edges, `build`, `open_graph` in `core/agent.py`**

Follow the spec's Nodes and Edges sections. The decisions the spec leaves open:
- `messages` uses `Annotated[list[dict], operator.add]`. Every other field overwrites. `route_intent` returns `messages: [user msg]` plus the per-turn resets: `retries=0, regenerated=False, unsupported=[], warning=None, queries_tried=[], documents=[], answer="", citations=[], graph=None, query=question`.
- `grade_documents` stores `relevant(documents)` back into `documents`. The router after it checks three cases: `documents` non-empty → generate; the *raw* retrieval was empty (keep a `found: int` count from retrieve in state) → not_found; otherwise retries < 2 → rewrite_query, else not_found.
- `generate` builds `Hit(**d)` from each dict to reuse `query._context`.
- `self_check` normalises the reply to `[x for x in reply.get("unsupported", []) if isinstance(x, str) and x.strip()]` when the value is a list, otherwise `[]`. It catches `ValueError` from `complete_json` and logs it with `logging.getLogger(__name__).warning`. The router after it: empty → respond; not `regenerated` → generate (the node sets `regenerated=True` on this path); else respond with `warning = f"{WARNING} " + "; ".join(unsupported)`.
- The `rewrite_query` prompt includes the last 4 entries of `messages` as `role: content` lines.
- `not_found` answer: `f"{NOT_FOUND} Searched for: " + ", ".join(repr(q) for q in queries_tried)`; with zero queries tried it's just `NOT_FOUND`.
- `EXPLAIN_SYSTEM = CITE_RULES + "Explain step by step for a student, defining terms before using them.\n" + "If the context does not contain the answer, reply exactly: This isn't in your material."`

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest -q tests/test_agent.py`
Expected: PASS. Then run the full suite: `.venv/Scripts/python -m pytest -q`, all pass.

- [ ] **Step 5: Commit**

```bash
git add core/agent.py tests/test_agent.py
git commit -m "feat: agent graph with corrective retrieval, self-check and sqlite checkpoints"
```

---

### Task 4: `/agent` streaming endpoint and history

**Files:**
- Modify: `api/main.py` (lifespan; new `AgentIn` body; new section "Agent" after Q&A)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `agent.open_graph`, `agent.LABELS`, graph input and state keys (Task 3).
- Produces:
  - `POST /agent` with body `AgentIn{message: str (1..2000), thread_id: str (1..100), topic_id: Optional[str], intent: Optional[Literal["qa","explain","connect","quiz","review"]], topics: Optional[list[str]] (max 2)}`. It returns NDJSON lines:
    - `{"event":"node","node":n,"label":LABELS[n]}`, plus `"query"` for `retrieve` / `rewrite_query`
    - `{"event":"done","intent","answer","citations","warning","graph"}`
    - or `{"event":"error","error":"Type: msg"}`
  - `GET /agent/{thread_id}` → `{"messages": [...]}`.
  - `app.state.agent` is the compiled graph.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py
import json
from core import agent, config

@pytest.fixture
def agent_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "agent_db_path", (tmp_path / "agent.db").as_posix())
    monkeypatch.setattr(api.main, "UPLOAD_DIR", tmp_path / "uploads")
    with TestClient(api.main.app) as c:
        yield c

def _events(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]

def test_agent_streams_nodes_then_done(agent_client, monkeypatch):
    async def search(q, tid): return [{"chunk_id": 1, "source_id": 1, "source_title": "N", "page": 1,
                                       "section": None, "text": "t", "score": 2.0}]
    async def complete(p, system=None, **kw): return "A [1]."
    monkeypatch.setattr(agent, "search_material", search)
    monkeypatch.setattr(llm, "complete", complete)
    ev = _events(agent_client.post("/agent", json={"message": "what is t", "thread_id": "x"}))
    assert [e["node"] for e in ev if e["event"] == "node"][:2] == ["route_intent", "retrieve"]
    assert ev[1]["label"] == "searching" and ev[1]["query"] == "what is t"
    assert ev[-1]["event"] == "done" and ev[-1]["answer"] == "A [1]." and ev[-1]["citations"]
    msgs = agent_client.get("/agent/x").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]

def test_agent_error_event_then_next_turn_works(agent_client, monkeypatch):   # Review Focus 2
    async def boom(q, tid): raise RuntimeError("ollama down")
    monkeypatch.setattr(agent, "search_material", boom)
    ev = _events(agent_client.post("/agent", json={"message": "q", "thread_id": "y"}))
    assert ev[-1] == {"event": "error", "error": "RuntimeError: ollama down"}
    ev = _events(agent_client.post("/agent", json={"message": "quiz me", "thread_id": "y"}))
    assert ev[-1]["event"] == "done" and ev[-1]["answer"] == agent.DEFERRED

def test_agent_history_unknown_thread_is_empty(agent_client):
    assert agent_client.get("/agent/nope").json() == {"messages": []}
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/Scripts/python -m pytest -q tests/test_api.py -k agent`
Expected: 404 / 405 failures.

- [ ] **Step 3: Implement it**

In `lifespan`, `async with agent.open_graph() as graph: app.state.agent = graph` wraps the existing `yield`, so teardown order is unchanged. The endpoint is an async generator over `app.state.agent.astream(..., stream_mode="updates")`. Each update dict is `{node: partial_state}`; the `query` comes from `partial_state.get("query")`. After the loop, it reads `aget_state(cfg).values` for `done`. The whole loop sits in `try/except Exception as exc` that yields the error line. It returns `StreamingResponse(gen(), media_type="application/x-ndjson")`, with each line `json.dumps(obj) + "\n"`.
The existing `client` fixture must also point `agent_db_path` at `tmp_path`. Add the monkeypatch line there too, so no test writes `data/agent.db`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: /agent NDJSON stream of node events and thread history"
```

---

### Task 5: Agent tab in the UI, README, real-model check

**Files:**
- Modify: `ui/app.py` (the tabs list near line 454; a new `with tab_agent:` block after Ingest; module docstring "3 tabs" → "4 tabs")
- Modify: `README.md` (Phase 3 notes, matching the Phase 2 notes section: what the agent does, `RELEVANCE_MIN`, `AGENT_DB_PATH`, `data/agent.db`)

**Interfaces:**
- Consumes: `POST /agent`, `GET /agent/{thread_id}` (Task 4); `_render_citations(citations)` in `ui/app.py`; `st.session_state.selected_topic` (`"all_topics"` → `None`).

- [ ] **Step 1: Implement the tab**

Tabs: `["📥 Ingest Material", "🤖 Agent", "💬 Ask Questions", "🔗 Connect Topics"]`.
- `session_state.agent_thread_id = str(uuid.uuid4())` if missing. `session_state.agent_messages` is loaded once per thread from `GET /agent/{id}` via `api()`.
- "New conversation" button: new uuid, clear `agent_messages`, `st.rerun()`.
- An optional `st.multiselect("Connect with…", topic names, max_selections=2)` sends `topics` only when 2 are chosen.
- On `st.chat_input(..., key="agent_input")`: inside `st.status("Thinking…", expanded=True)`, use `with httpx.stream("POST", f"{settings.api_url}/agent", json=..., timeout=900) as r:` and `for line in r.iter_lines()`. For a node event, write `label` (plus `→ "query"` when present) and update the status label. For `done`, set the status to complete and render the answer, `st.warning(warning)`, `_render_citations`, and the graph via `st.components.v1.html(graph, height=500)`. For `error`, set the status to error and `st.error`. On a `httpx.TransportError`, show the same "Cannot reach the StudyMind API" message as `api()`.
- Append both messages to `agent_messages` so the history renders on rerun.

- [ ] **Step 2: Smoke-check the UI wiring offline**

Run: `.venv/Scripts/python -m py_compile ui/app.py` (prints nothing), then the full test suite (all pass).

- [ ] **Step 3: Real-model check (done-when 1), by hand with Ollama running**

Use a throwaway environment: `DATABASE_URL=sqlite:///<scratch>/s.db QDRANT_PATH=<scratch>/q AGENT_DB_PATH=<scratch>/a.db`. Start the API and UI, ingest one PDF into a topic, then ask a vague question (for example "that thing from the second lecture?").
Pass: the status box shows `searching` → `checking sources` → `rewriting query → "<specific query>"` → `searching` → … → answer or the honest not-found. Also ask one clear question and check that it answers without a rewrite in about 30 s.
If a clearly related question gets rewritten, record its top score. The cutoff may need tuning, so report back rather than changing `-3.0` silently.

- [ ] **Step 4: Commit**

```bash
git add ui/app.py README.md
git commit -m "feat: Agent chat tab streaming the agent's steps; README phase 3 notes"
```
