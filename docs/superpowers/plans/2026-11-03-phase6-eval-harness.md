# Phase 6 — Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure every retrieval/generation design choice (configs B→H + E-nomic) on a gold-labelled dataset and publish a README results table + Mermaid chart, with a judge validated against hand labels.

**Architecture:** Three stages over one JSON file per config: `eval.run` (retrieve + answer, records hits/answer/latency/usage), `eval.judge` (adds faithfulness/correctness, cached), `eval.report` (recomputes metrics, rewrites a marked README section). Gold is anchored by `{source, page, quote}` / figure-on-page so it survives re-ingest and both chunkers. Two corpora (fixed, structure) live in `data/eval/<corpus>/`, isolated from the app's data.

**Tech Stack:** Python 3.13, existing core/ (SQLModel, embedded Qdrant, fastembed, LiteLLM → Ollama qwen2.5:7b, LangGraph), pytest + pytest-asyncio. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-03-phase6-eval-harness-design.md`

## Global Constraints

- No new dependencies (no RAGAS, matplotlib, scikit-learn).
- Tests are offline, use `tests/conftest.py` fakes; fakes of `llm.complete` / `complete_json` must accept `system` positionally.
- Any test that lets code mutate `settings.rerank`, `settings.figures`, `settings.query_rewrite`, `settings.retriever`, `settings.chunker`, `settings.judge_model`, `settings.agent_db_path` must first `monkeypatch.setattr(settings, <name>, <current value>)` so it is restored.
- All files read/written with `encoding="utf-8"`; CLIs call `sys.stdout.reconfigure(encoding="utf-8")` (console is cp1252).
- Refusal = answer `.strip().startswith("This isn't in your material")`.
- Quote ≤ 40 words; Recall cut-off k = 5; retrieval k = 6; κ threshold 0.6; calibration n = 30; judge prompt version `1`.
- App data (`data/studymind.db`, `data/qdrant`, `data/agent.db`) is never opened by eval code; corpora live under `data/eval/<corpus>/`.
- Run tests: `.venv/Scripts/python -m pytest -q -p no:cacheprovider <path>`; the full suite (258+ tests) goes to a log file or background.
- One commit per task; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Edit Python with the Edit/Write tools, never shell heredocs.

## Review Focus

1. **Dataset gold names a source the corpus doesn't have** (typo, renamed PDF) → Recall would silently read 0. Expected: `eval.run` fails fast listing the unknown titles. Test in Task 3.
2. **Rows that errored** (Ollama down mid-run) → expected: excluded from every metric and shown as "errors: n" in the report, never counted as misses. Test in Task 5.
3. **Agent refusal carries a suffix** (`This isn't in your material. Searched for: ...`) → expected: `refused = True`. Test in Task 3.
4. **Calibration interrupted** (Ctrl-C after 12 labels) → expected: those 12 are already on disk and skipped on resume. Test in Task 4.
5. **README with CRLF line endings / no markers yet** → expected: marker section appended once; on rerun replaced, rest byte-identical. Test in Task 5.

---

### Task 1: Metrics

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `normalise(text: str) -> str` — lowercase, `-\n` joined, all whitespace runs → single space, stripped.
  - `matches(hit: dict, gold: dict) -> bool` — `hit` has keys `source_title, page, kind, text`; `gold` is `{source, page, quote}` or `{source, page, kind: "figure"}`.
  - `recall_at(hits: list[dict], gold: list[dict], k: int = 5) -> float`
  - `mrr(hits: list[dict], gold: list[dict]) -> float`
  - `citation_accuracy(answer: str, hits: list[dict], gold: list[dict]) -> float | None` — `[n]` is 1-based into `hits`; out-of-range markers count as wrong.
  - `cohen_kappa(a: list, b: list) -> float` — returns 1.0 when expected agreement is 1.
  - `is_refusal(answer: str) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
from eval import metrics as m

T = lambda title, page, text, kind="text": {"source_title": title, "page": page, "text": text, "kind": kind}
Q = {"source": "ODS", "page": 3, "quote": "A skiplist is a sequence"}
F = {"source": "ODS", "page": 9, "kind": "figure"}

def test_normalise_joins_hyphen_breaks_and_whitespace():
    assert m.normalise("Skip-\nList  is\n a  LIST") == "skiplist is a list"

def test_quote_match_needs_same_source_and_contained_quote():
    assert m.matches(T("ODS", 4, "Intro. A  skiplist is a\nsequence of lists."), Q)  # page not required for quotes
    assert not m.matches(T("Other", 3, "A skiplist is a sequence"), Q)
    assert not m.matches(T("ODS", 3, "A skiplist is not a sequence"), Q)

def test_figure_gold_matches_figure_hit_on_page_only():
    assert m.matches(T("ODS", 9, "any caption", "figure"), F)
    assert not m.matches(T("ODS", 9, "any caption", "text"), F)
    assert not m.matches(T("ODS", 8, "any caption", "figure"), F)

def test_recall_and_mrr():
    hits = [T("ODS", 1, "x"), T("ODS", 9, "c", "figure"), T("ODS", 3, "a skiplist is a sequence of")]
    assert m.recall_at(hits, [Q, F]) == 1.0
    assert m.recall_at(hits, [Q, F], k=2) == 0.5
    assert m.mrr(hits, [Q, F]) == 0.5
    assert m.mrr([T("ODS", 1, "x")], [Q]) == 0.0
    assert m.recall_at([], [Q]) == 0.0

def test_citation_accuracy():
    hits = [T("ODS", 3, "a skiplist is a sequence"), T("ODS", 1, "x")]
    assert m.citation_accuracy("Yes [1] and [2].", hits, [Q]) == 0.5
    assert m.citation_accuracy("Yes [7].", hits, [Q]) == 0.0
    assert m.citation_accuracy("No citations.", hits, [Q]) is None

def test_kappa():
    assert m.cohen_kappa(["c", "w", "c", "w"], ["c", "w", "c", "w"]) == 1.0
    assert round(m.cohen_kappa(["c", "c", "w", "w"], ["c", "w", "c", "w"]), 3) == 0.0
    assert m.cohen_kappa(["c", "c"], ["c", "c"]) == 1.0

def test_is_refusal():
    assert m.is_refusal("  This isn't in your material. Searched for: 'x'")
    assert not m.is_refusal("It is in your material [1].")
```

- [ ] **Step 2: Run to verify failure** — `.venv/Scripts/python -m pytest -q -p no:cacheprovider tests/test_eval_metrics.py` → FAIL (`cannot import name 'metrics'`).
- [ ] **Step 3: Implement `eval/metrics.py`** with the signatures above. Plain Python; citation markers via `re.findall(r"\[(\d+)\]", answer)`.
- [ ] **Step 4: Run the test again** → all pass.
- [ ] **Step 5: Commit** — `feat(eval): retrieval and agreement metrics`.

---

### Task 2: Retrieval switches, judge model setting, LLM usage tally

**Files:**
- Modify: `core/config.py` (Settings, retrieval block), `core/retrieval.py` (`relevant`, `HybridRetriever._search`, `DenseRetriever._search`), `core/llm.py` (`LLMClient`)
- Test: `tests/test_retrieval.py`, `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `settings.rerank: bool = True`, `settings.figures: bool = True`, `settings.judge_model: str = ""` (empty → `llm_model`).
  - `HybridRetriever` with `rerank=False`: returns the top k of the RRF order, `Hit.score` = RRF score; `retrieval.rerank` is not called.
  - `relevant(hits)` keeps every hit when `settings.retriever == "dense"` **or** `not settings.rerank`.
  - With `figures=False`, both retrievers drop hits whose `media["kind"] == "figure"` before cutting to k (Dense queries `limit=k + 10` so k usually remain).
  - `llm.usage: dict` with keys `calls, prompt_tokens, completion_tokens, cost_usd`, incremented in `complete` after each successful provider call (None token counts add 0).

- [ ] **Step 1: Write the failing tests** (in `tests/test_retrieval.py`; plant one text chunk and one figure chunk on the same topic with `index_chunks(..., figure={"image_path": "x.png", "bbox": "0,0,1,1"})` as the existing test at line ~252 does)

```python
@pytest.mark.asyncio
async def test_rerank_off_skips_cross_encoder_and_keeps_all_relevant(monkeypatch):
    monkeypatch.setattr(settings, "rerank", False)
    monkeypatch.setattr(retrieval, "rerank", lambda q, t: (_ for _ in ()).throw(AssertionError("reranked")))
    # plant 3 chunks mentioning "heap"
    hits = await retrieval.HybridRetriever().retrieve("heap", tid)
    assert hits and all(h.score < 1 for h in hits)          # RRF scores, not rerank logits
    assert retrieval.relevant(hits) == hits

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["hybrid", "dense"])
async def test_figures_off_drops_figure_hits(monkeypatch, kind):
    monkeypatch.setattr(settings, "retriever", kind)
    # plant text chunk "heap insert is log n" and figure chunk "Figure: heap insert diagram"
    assert any((h.media or {}).get("kind") == "figure" for h in await retrieval.base_retriever().retrieve("heap insert", tid))
    monkeypatch.setattr(settings, "figures", False)
    hits = await retrieval.base_retriever().retrieve("heap insert", tid)
    assert hits and not any((h.media or {}).get("kind") == "figure" for h in hits)
```

In `tests/test_llm.py`, following its existing `fake_acompletion` pattern (returns an object with `usage.prompt_tokens=5, completion_tokens=2`):

```python
@pytest.mark.asyncio
async def test_usage_tally_counts_calls_and_tokens(monkeypatch):
    monkeypatch.setattr(llm, "usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
    await llm.complete("hi"); await llm.complete("again")
    assert llm.usage == {"calls": 2, "prompt_tokens": 10, "completion_tokens": 4, "cost_usd": 0.0}
```

- [ ] **Step 2: Run to verify failure** — `tests/test_retrieval.py tests/test_llm.py` → the 3 new tests FAIL (`no attribute 'rerank'` on settings / `usage`).
- [ ] **Step 3: Implement** the settings, the two retriever branches, `relevant` change, and `self.usage` (initialised in `LLMClient.__init__`).
- [ ] **Step 4: Run `tests/test_retrieval.py tests/test_llm.py tests/test_agent.py tests/test_rewrite.py`** → all pass.
- [ ] **Step 5: Commit** — `feat: rerank/figures switches, judge_model setting, llm usage tally`.

---

### Task 3: Configs, corpora and the `--config` runner

**Files:**
- Create: `eval/configs.py`
- Modify: `eval/run.py` (rewrite), `core/query.py:110-129` (split `_answer`), `eval/dataset.jsonl` (empty it — delete the 3 EXAMPLE rows)
- Test: `tests/test_eval.py` (replace both old tests)

**Interfaces:**
- Consumes: Task 1 `is_refusal`; Task 2 settings and `llm.usage`.
- Produces:
  - `core.query._generate(question: str, hits: list[Hit]) -> tuple[str, list[dict]]` — (answer, citations); `_answer` calls it, behaviour unchanged.
  - `eval.configs.CORPUS_ROOT = DATA_DIR / "eval"`, `BEST_F = "F-multi"` (edited after F runs).
  - `eval.configs.CONFIGS: dict[str, dict]` — each value has `change: str`, `corpus: "fixed"|"structure"`, `agent: bool` and settings overrides from {`retriever, rerank, figures, query_rewrite, embed_model`}; rows exactly as the spec's table (B–F, E-nomic with `figures=False`; G = `{**CONFIGS[BEST_F], figures: True}`; H = `{**G, agent: True}`). `ORDER = ["B","C","D","E","F-multi","F-hyde","G","H","E-nomic"]`.
  - `eval.configs.open_corpus(corpus: str) -> None` — sets `settings.database_url`, `qdrant_path`, `agent_db_path` to `CORPUS_ROOT/corpus/{studymind.db,qdrant,agent.db}` (mkdir), `settings.chunker = corpus`, then `db.reset_engine()`, `retrieval.reset_store()`, `init_db()`.
  - `eval.configs.apply(cfg: dict) -> None` — sets every override key present on `settings` (missing keys reset to defaults: `rerank=True, figures=True, query_rewrite="none"`).
  - `eval.run.run(name: str, dataset: Path, out_dir: Path, limit: int | None = None, corpus: bool = True) -> Path` — `corpus=False` skips `open_corpus` (tests). Writes `out_dir/<name>.json`:
    `{"config": {"name", **cfg}, "run_at": iso, "git_sha": str, "rows": [{id, type, question, hits: [{source_title, page, chunk_id, kind, score, text}], answer, refused, latency_s, prompt_tokens, completion_tokens, cost_usd, error}]}`.
    Before answering, raises `ValueError` naming any `gold[].source` not in `topics.list_sources()` titles.
    Agent rows: `open_graph()` (default path = `settings.agent_db_path`), input `{"messages": [], "question": q, "topics": [], "topic_id": row["topic_id"]}`, `thread_id = f"eval-{name}-{row['id']}-{uuid4().hex[:6]}"`; hits from final `state["documents"]`.
  - CLI: `python -m eval.run --config <name> [--limit N] [--dataset PATH]`; prints the output path; `retrieval.reset_store()` in `finally`.

- [ ] **Step 1: Write the failing tests** (`tests/test_eval.py`, replacing the file). Use `tests.planting.plant(["heaps support insert in log n time", "skiplists are layered lists"], {}, topic="ODS")` → source title `"ODS notes"`. Write a dataset row with `gold: [{"source": "ODS notes", "page": 1, "quote": "heaps support insert"}]`.

```python
async def test_run_writes_config_json(tmp_path, monkeypatch, planted_dataset):  # fixture builds the above
    for k in ("retriever", "rerank", "figures", "query_rewrite"): monkeypatch.setattr(settings, k, getattr(settings, k))
    out = await run.run("E", planted_dataset, tmp_path, corpus=False)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert out.name == "E.json" and doc["config"]["name"] == "E" and doc["config"]["corpus"] == "structure"
    row = doc["rows"][0]
    assert row["hits"][0]["source_title"] == "ODS notes" and row["hits"][0]["kind"] == "text"
    assert {"answer", "refused", "latency_s", "prompt_tokens", "cost_usd", "error"} <= row.keys()
    assert row["answer"] == "Transformer Models" and row["refused"] is False

async def test_unknown_gold_source_fails_fast(tmp_path, planted_dataset_with_bad_source):
    with pytest.raises(ValueError, match="Missing Book"):
        await run.run("E", planted_dataset_with_bad_source, tmp_path, corpus=False)

async def test_agent_config_records_suffixed_refusal(tmp_path, monkeypatch, planted_dataset):
    monkeypatch.setattr(settings, "agent_db_path", str(tmp_path / "agent.db"))
    monkeypatch.setattr(settings, "relevance_min", 99.0)       # nothing relevant → not_found node
    # also snapshot retriever/rerank/figures/query_rewrite as in the first test
    doc = json.loads((await run.run("H", planted_dataset, tmp_path, corpus=False)).read_text(encoding="utf-8"))
    assert doc["rows"][0]["refused"] is True and doc["rows"][0]["answer"].startswith("This isn't in your material. Searched")

def test_configs_ladder():
    C = configs.CONFIGS
    assert configs.ORDER == list(C)
    assert C["B"]["corpus"] == "fixed" and C["D"]["rerank"] is False
    assert C["G"]["figures"] is True and C["H"]["agent"] is True
    assert all(C[n]["figures"] is False for n in ("B", "C", "D", "E", "F-multi", "F-hyde", "E-nomic"))
```

- [ ] **Step 2: Run to verify failure** — `tests/test_eval.py` → FAIL (`configs` missing / `run()` signature).
- [ ] **Step 3: Implement** `_generate` split, `eval/configs.py`, `eval/run.py`. `git_sha` via `subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()`; usage deltas from `llm.usage` before/after each row; a row's exception is caught into `error` (the run continues).
- [ ] **Step 4: Run** `tests/test_eval.py tests/test_query.py tests/test_agent.py` → pass.
- [ ] **Step 5: Commit** — `feat(eval): named configs, isolated corpora, --config runner`.

---

### Task 4: Judge and calibration

**Files:**
- Create: `eval/judge.py`
- Test: `tests/test_eval_judge.py`

**Interfaces:**
- Consumes: Task 3 result-file shape; Task 1 `cohen_kappa`, `is_refusal`.
- Produces:
  - `PROMPT_VERSION = 1`; `CACHE = EVAL_DIR / "results" / "judge_cache.json"`; `LABELS = EVAL_DIR / "judge_labels.jsonl"`.
  - `context_text(hits: list[dict]) -> str` — `"[n] <title>, p.<page>\n<text>"` blocks.
  - `async faithfulness(context: str, answer: str) -> float | None` — JSON `{"claims": [{"claim", "supported"}]}`; supported/total; 0 claims → 1.0.
  - `async correctness(question: str, gold_answer: str, answer: str) -> float | None` — `{"verdict": "correct|partial|wrong"}` → 1/0.5/0; other verdicts → None.
  - Both go through `_ask(prompt, system) -> dict | None`: `llm.complete_json(prompt, system, model=settings.judge_model or settings.llm_model)`, one retry on `ValueError`, then None. Cache key `sha256(json.dumps([model, PROMPT_VERSION, kind, question, answer, context]))`.
  - `async judge_file(result_path: Path, dataset_path: Path, cache_path: Path = CACHE) -> dict` — writes `row["judge"] = {"faithfulness": f|None, "correctness": c|None}` (skip faithfulness for refusals and errored rows; skip correctness for `type == "unanswerable"` and errored rows); returns `{"judged": n, "failures": n}`.
  - `sample(results: dict[str, dict], labelled: set[tuple[str, str]], n: int = 30) -> list[tuple[str, dict]]` — round-robin over (type, judge verdict) buckets, deterministic (`random.Random(0)`), excludes `(config, id)` already labelled and rows without judge scores.
  - `calibrate(results, dataset, labels_path=LABELS, n=30, ask=input) -> None` — per item prints question/gold/answer/context, asks `correct? [c/p/w]` and `all claims supported? [y/n]`, **appends each label to `labels_path` immediately** as `{"config", "id", "correct": "c|p|w", "faithful": bool}`.
  - `agreement(labels: list[dict], results: dict[str, dict]) -> dict` — `{"n", "correctness": {"agree", "kappa"}, "faithfulness": {"agree", "kappa"}}`; judge correctness 1/0.5/0 → c/p/w; judge faithfulness `== 1` → True.
  - CLI: `python -m eval.judge <config>` | `python -m eval.judge --calibrate` (loads all results, prints agreement after labelling).

- [ ] **Step 1: Write the failing tests**

```python
async def test_faithfulness_and_correctness_parse(monkeypatch, judge_replies):
    judge_replies([{"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]},
                   {"verdict": "partial"}])
    assert await judge.faithfulness("[1] x", "a. b.") == 0.5
    assert await judge.correctness("q", "gold", "ans") == 0.5

async def test_retry_then_null(monkeypatch, judge_replies):
    judge_replies([ValueError("no json"), ValueError("no json")])
    assert await judge.correctness("q", "g", "a") is None

async def test_judge_file_uses_cache_and_skips_refusals(tmp_path, judge_replies, result_file):
    # result_file: rows q1 (answered, type text) and q2 (refused, type unanswerable)
    calls = judge_replies([{"claims": []}, {"verdict": "correct"}])
    stats = await judge.judge_file(result_file, dataset, tmp_path / "cache.json")
    rows = json.loads(result_file.read_text(encoding="utf-8"))["rows"]
    assert rows[0]["judge"] == {"faithfulness": 1.0, "correctness": 1.0}
    assert rows[1]["judge"] == {"faithfulness": None, "correctness": None}
    assert stats == {"judged": 1, "failures": 0}
    await judge.judge_file(result_file, dataset, tmp_path / "cache.json")
    assert len(calls) == 2                                     # second pass fully cached

def test_calibrate_writes_each_label_immediately(tmp_path, results_with_3_judged_rows):
    answers = iter(["c", "y", "w"])
    def ask(_):
        try: return next(answers)
        except StopIteration: raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        judge.calibrate(results_with_3_judged_rows, dataset, tmp_path / "labels.jsonl", n=3, ask=ask)
    lines = (tmp_path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["correct"] == "c"

def test_agreement():
    results = {"E": {"rows": [{"id": "q1", "judge": {"correctness": 1.0, "faithfulness": 1.0}},
                              {"id": "q2", "judge": {"correctness": 0.0, "faithfulness": 0.5}}]}}
    labels = [{"config": "E", "id": "q1", "correct": "c", "faithful": True},
              {"config": "E", "id": "q2", "correct": "c", "faithful": False}]
    a = judge.agreement(labels, results)
    assert a["n"] == 2 and a["correctness"]["agree"] == 0.5 and a["faithfulness"]["agree"] == 1.0
```

(`judge_replies` fixture: monkeypatches `llm.complete_json` with `async def f(prompt, system=None, **kw)` popping replies — raising those that are exceptions — and returns the call list.)

- [ ] **Step 2: Run to verify failure** — `tests/test_eval_judge.py` → FAIL (no module).
- [ ] **Step 3: Implement `eval/judge.py`.** System prompts: faithfulness — "You check whether each claim in an answer is supported by the numbered passages. Split the answer into atomic factual claims (skip citations and filler)."; correctness — "You compare a student-assistant answer with a reference answer. correct = same key facts; partial = some key facts or minor errors; wrong = missing or contradicting the key facts."
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** — `feat(eval): LLM judge with cache and hand-label calibration`.

---

### Task 5: Report → README

**Files:**
- Create: `eval/report.py`
- Modify: `README.md` (add an "Evaluation" heading with empty `<!-- eval:start -->` / `<!-- eval:end -->` markers — done by `update_readme`, not by hand)
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: Task 1 metrics; Task 3 `configs.CONFIGS`, `ORDER`; Task 4 `agreement`, `LABELS`.
- Produces:
  - `summarise(result: dict, dataset: dict[str, dict]) -> dict` — over non-error rows: `recall5, mrr` (answerable rows), `faithfulness, correctness` (mean of non-null judge scores, None if none), `refusal` (share refused among unanswerable rows), `latency, cost` (means), `errors, judge_failures` (counts), plus `diagram: {recall5, correctness}` over `type == "diagram"` rows. Means over empty sets are None.
  - `build(results: dict[str, dict], dataset: dict[str, dict], labels: list[dict]) -> str` — markdown: dataset counts by type; judge line (`Judge (qwen2.5:7b) vs 30 hand labels: correctness 83% agree, κ 0.71; faithfulness ...`, or `judge unvalidated` when no labels or any κ < 0.6 — then the Faithfulness/Correctness headers get ` (unvalidated)`); main table in `ORDER` with row A first (`A | Cognee baseline | not measured (Cognee removed in Phase 1)`), percentages to 1 decimal, None → `—`, errors noted as `(errors: n)` after the config name; diagram sub-table for configs with diagram rows; one ```` ```mermaid ```` `xychart-beta` bar chart of Recall@5 (%) by config.
  - `update_readme(readme: Path, section: str) -> None` — replaces text between the markers (inclusive markers kept); if markers absent, appends `\n## Evaluation\n\n<!-- eval:start -->\n...\n<!-- eval:end -->\n`. Reads/writes bytes-preserving newlines (`newline=""`).
  - CLI: `python -m eval.report` (loads `eval/results/*.json` except `judge_cache.json`).

- [ ] **Step 1: Write the failing tests**

```python
def test_summarise_excludes_errors_and_unanswerable_from_recall(dataset, result_E):
    # result_E rows: q1 text hit-gold-at-rank-1, q2 text error, q3 unanswerable refused
    s = report.summarise(result_E, dataset)
    assert s["recall5"] == 1.0 and s["mrr"] == 1.0 and s["errors"] == 1 and s["refusal"] == 1.0

def test_build_table_rows_and_chart(dataset, result_E):
    md = report.build({"E": result_E}, dataset, labels=[])
    assert "| A | Cognee baseline | not measured (Cognee removed in Phase 1)" in md
    assert "| E (errors: 1) |" in md and "100.0" in md
    assert "judge unvalidated" in md and "```mermaid" in md and "xychart-beta" in md

def test_update_readme_appends_then_replaces_preserving_crlf(tmp_path):
    p = tmp_path / "README.md"; p.write_bytes(b"# StudyMind\r\n\r\nIntro.\r\n")
    report.update_readme(p, "one")
    report.update_readme(p, "two")
    text = p.read_bytes()
    assert text.startswith(b"# StudyMind\r\n\r\nIntro.\r\n") and text.count(b"<!-- eval:start -->") == 1
    assert b"two" in text and b"one" not in text
```

- [ ] **Step 2: Run to verify failure** — FAIL (no module).
- [ ] **Step 3: Implement `eval/report.py`.**
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** — `feat(eval): results report with README table and Mermaid chart`.

---

### Task 6: Labeller

**Files:**
- Create: `eval/label.py`
- Test: `tests/test_eval_label.py`

**Interfaces:**
- Consumes: Task 1 `normalise`; Task 3 `open_corpus`; `retrieval.get_retriever`, `core.db.Chunk/Source/ChunkMedia`.
- Produces:
  - `next_id(rows: list[dict]) -> str` — `q001`-style, max existing + 1.
  - `first_quote(text: str) -> str | None` — first sentence (split on `(?<=[.!?])\s+`) with 5–40 words, else the first 40 words if ≥ 5, else None.
  - `async draft(chunk_text: str, kind: str) -> dict | None` — `llm.complete_json` → `{question, gold_answer, quote}`; for `kind == "text"` the quote must satisfy `normalise(quote) in normalise(chunk_text)` and be ≤ 40 words, one re-ask, else None. For figures the quote is ignored.
  - `async propose(n: int, qtype: str, dataset_path: Path, ask=input) -> int` — samples unused chunks (figure chunks for `diagram`, text otherwise; spread by `(source_id, page)`; `random.Random(0)`), shows chunk + draft, `[a]ccept / [e]dit / [s]kip / [q]uit`; edit re-prompts each field (empty keeps). Appends accepted rows (`topic_id` from the chunk, gold from source title + page + quote or figure) immediately; returns count accepted.
  - `async find(question: str, qtype: str, dataset_path: Path, ask=input) -> dict` — prints top 10 hits (`get_retriever(rewrite=False).retrieve(question, None, k=10)`), asks for space-separated hit numbers (empty → `gold: []`), builds gold (figure hit → figure gold; text hit → `first_quote`), asks for the gold answer (unanswerable: default `"Not in the material."`), appends and returns the row.
  - CLI: `python -m eval.label propose --n 20 --type text|diagram` | `python -m eval.label find "<question>" --type text|multihop|cross|unanswerable` (both `open_corpus("structure")` first).

- [ ] **Step 1: Write the failing tests**

```python
def test_next_id_and_first_quote():
    assert label.next_id([]) == "q001" and label.next_id([{"id": "q009"}, {"id": "q002"}]) == "q010"
    assert label.first_quote("Hi. A heap is a tree with the heap property at every node.") == "A heap is a tree with the heap property at every node."
    assert label.first_quote("Too short.") is None

async def test_draft_rejects_non_verbatim_quote_after_one_retry(monkeypatch):
    replies = [{"question": "q", "gold_answer": "g", "quote": "invented words here now"}] * 2
    monkeypatch.setattr(llm, "complete_json", _popper(replies))
    assert await label.draft("Heaps support insert in log n time.", "text") is None

async def test_propose_accept_skip_appends(tmp_path, monkeypatch):
    # plant 2 text chunks; complete_json returns verbatim quotes for each
    path = tmp_path / "d.jsonl"
    n = await label.propose(2, "text", path, ask=_answers(["a", "s"]))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert n == 1 and rows[0]["id"] == "q001" and rows[0]["gold"][0]["quote"] and rows[0]["type"] == "text"

async def test_find_unanswerable_has_empty_gold(tmp_path):
    row = await label.find("who invented treaps", "unanswerable", tmp_path / "d.jsonl", ask=_answers(["", ""]))
    assert row["gold"] == [] and row["gold_answer"] == "Not in the material."
```

- [ ] **Step 2: Run to verify failure** — FAIL (no module).
- [ ] **Step 3: Implement `eval/label.py`.**
- [ ] **Step 4: Run** → pass; then full suite to a log: `.venv/Scripts/python -m pytest -q -p no:cacheprovider > "$TEMP/p6-suite.log" 2>&1`, expect 0 failures.
- [ ] **Step 5: Commit** — `feat(eval): labelling CLI (propose and find)`.

---

### Task 7: Corpus fetch and ingest

**Files:**
- Create: `eval/ingest.py`, `eval/corpus/README.md`
- Modify: `.gitignore` (add `eval/corpus/*.pdf`, `data/eval/`)
- Test: `tests/test_eval_ingest.py`

**Interfaces:**
- Consumes: Task 3 `open_corpus`; `core.ingest.ingest_pdf`, `core.figures.caption_pending`, `core.topics.create_topic/list_sources`.
- Produces:
  - `SOURCE_URL = "https://opendatastructures.org/ods-python.pdf"`, `CHAPTERS: dict[str, tuple[int, int]]` — filename → 1-based inclusive page range, 3 chapters with figures (filled in Step 4 after inspecting the PDF's outline).
  - `fetch(dest: Path) -> list[Path]` — downloads once to `dest/ods-python.pdf` (skip if present), writes one PDF per chapter via `pymupdf` `insert_pdf(from_page, to_page)`; skips chapters already split.
  - `async build(corpus: str, pdf_dir: Path, topic: str = "ods") -> dict` — `open_corpus(corpus)`, ensures topic `ods` exists, ingests each PDF whose stem isn't already a source title, then loops `caption_pending()` until it returns 0 with no `PendingFigure` rows left; returns `{"sources": n, "chunks": n, "figures": n}`.
  - CLI: `python -m eval.ingest [--corpus fixed|structure|both]` (default both; fetches first).

- [ ] **Step 1: Write the failing test** — with `open_corpus` monkeypatched to only set `settings.chunker`, and a synthetic PDF from `tests/pdfs.py` in `tmp_path`:

```python
async def test_build_is_idempotent(tmp_path, monkeypatch):
    stats = await eval_ingest.build("structure", pdf_dir)
    again = await eval_ingest.build("structure", pdf_dir)
    assert stats["sources"] == 1 and again["sources"] == 1 and stats["chunks"] == again["chunks"] > 0
```

- [ ] **Step 2: Run to verify failure**; **Step 3: implement**; **Step 4: run** → pass.
- [ ] **Step 5: Real fetch.** Check the licence statement on opendatastructures.org (expect CC BY 2.5) and write it, the URL and the chapter page ranges into `eval/corpus/README.md`. Inspect `pymupdf.open(...).get_toc()` and pick 3 chapters with figures (target: Linked Lists, Skiplists, Binary Trees). Run `python -m eval.ingest` (background, log to scratchpad). Expected: both corpora report sources = 3, chunks > 0, figures > 0. Spot-check 3 figure captions in `data/eval/structure/studymind.db` for sanity; if figures = 0, stop and report.
- [ ] **Step 6: Commit** — `feat(eval): fetch and ingest Open Data Structures corpora`.

---

### Task 8: Run the evaluation (operator task; user at keyboard for 8b and 8e)

- [ ] **8a.** Ollama up (`qwen2.5:7b`), corpora built.
- [ ] **8b. User labels** ~20 questions: `python -m eval.label propose --n 8 --type text`, `propose --n 4 --type diagram`, then 3 multihop and 5 unanswerable via `find`. Commit `eval/dataset.jsonl`.
- [ ] **8c.** Run `B C D E F-multi F-hyde E-nomic` (each `python -m eval.run --config X` then `python -m eval.judge X`, background, logs to scratchpad). Set `BEST_F` to the higher Recall@5 of F-multi/F-hyde (tie → F-multi); run and judge `G`, then `H`.
- [ ] **8d.** `python -m eval.report`; inspect the table for errors/judge failures; rerun any config with errors.
- [ ] **8e. User calibrates:** `python -m eval.judge --calibrate` (30 labels). Rerun `python -m eval.report`.
- [ ] **8f.** Commit `eval/results/*.json`, `eval/judge_labels.jsonl`, `eval/configs.py` (BEST_F), `README.md` — `eval: starter results (n=~20) for configs B–H`.
