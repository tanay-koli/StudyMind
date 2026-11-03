# Phase 6 — Evaluation harness: design

Date: 2026-11-03 · Branch: `phase-6` · Master plan: "StudyMind v2 — Implementation Plan.pdf", Phase 6 + "Evaluation plan".

## Goal

Turn each design choice (chunker, hybrid, reranker, query rewriting, figure captions, corrective agent loop,
embedding model) into a measured comparison you can defend. **Done when:** the README has a results table with
real numbers and one chart, and the judge's agreement with hand labels is reported next to it.

## Decisions (agreed in brainstorming)

| Question | Decision |
|---|---|
| Material | Starter set on an openly licensed textbook (*Open Data Structures*, CC BY); your own course replaces it later. |
| Starter size | ~20 questions now (harness first), growing toward the plan's 150 later. |
| Labelling | LLM proposes question + gold answer + verbatim quote from a chunk; you accept / edit / skip. A second `find` mode lets you write a question and tick gold among the top 10 hits. |
| Gold anchoring | `{source, page, quote}` for text, `{source, page, kind: "figure"}` for figures; never raw chunk ids. |
| Judge | Own prompts on local qwen2.5:7b via `settings.judge_model`; no RAGAS. Hosted judge = one `.env` line. |
| Judge validation | 30 hand labels, % agreement + Cohen's κ; κ < 0.6 marks judged columns "judge unvalidated". |
| Ablations | Full ladder B→H plus an `E-nomic` side row. Row A reported as "not measured (Cognee removed in Phase 1)". |
| Chart | Mermaid `xychart-beta` block in the README, generated; no new dependency. |
| Pipeline | Separate stages: `run` → `judge` → `report`, each reading/writing `eval/results/<config>.json`. |

## Components

### 1. Settings switches (core/)

- `settings.rerank: bool = True` — `HybridRetriever._search` returns the RRF order (top k, score = RRF score)
  without the cross-encoder when False. With `rerank=False`, `relevant()` treats scores like the dense retriever's
  (keeps all hits), because RRF scores are not reranker logits.
- `settings.figures: bool = True` — when False, `HybridRetriever` and `DenseRetriever` drop hits whose
  media kind is `figure` before cutting to k (they fetch k + spare candidates so k remain where possible).
- `settings.judge_model: str = ""` — empty means use `llm_model`.
- Both retrieval switches live in the base retrievers, which `RewritingRetriever` and the agent's
  `search_material` (via `get_retriever(rewrite=False)`) both wrap, so they apply on every path.
- `core/llm.py` gains a process-wide usage tally (`llm.usage`: calls, prompt_tokens, completion_tokens,
  cost_usd) incremented beside the existing log line, so the runner can read per-question deltas.

### 2. Configs — `eval/configs.py`

One dict, name → overrides; each row is the previous best plus one change:

| Name | corpus | Overrides |
|---|---|---|
| B | fixed | retriever=dense, figures=False |
| C | structure | retriever=dense, figures=False |
| D | structure | retriever=hybrid, rerank=False, figures=False |
| E | structure | retriever=hybrid, figures=False |
| F-multi | structure | E + query_rewrite=multi |
| F-hyde | structure | E + query_rewrite=hyde |
| G | structure | best F (by Recall@5; chosen by editing one line after F runs) + figures=True |
| H | structure | G + agent=True |
| E-nomic | structure | E + embed_model=nomic |

Note: the agent never uses the rewriting retriever (it rewrites itself in `rewrite_query`), so H's
`query_rewrite` has no effect; H measures grade/retry/self-check against G.

### 3. Corpora — `python -m eval.ingest`

The chunker runs at ingest time, so the same PDFs are ingested twice: `data/eval/fixed/` and
`data/eval/structure/`, each with its own `studymind.db` and `qdrant/` (via `DATABASE_URL` / `QDRANT_PATH`
overrides applied in-process before `init_db`). Source PDFs live in `eval/corpus/*.pdf` (licence note in
`eval/corpus/README.md`). Figure captioning runs for both corpora, waiting for the caption queue to drain. The topic is
`ods` (one topic; cross-topic questions are out of the starter set). The nomic collection is built by
`ensure_indexed` on first use. Your app's `data/studymind.db` is never touched.

### 4. Dataset — `eval/dataset.jsonl`

```json
{"id": "q017", "type": "text", "topic_id": "ods",
 "question": "...", "gold_answer": "...",
 "gold": [{"source": "<source title>", "page": 87, "quote": "a skiplist is a sequence of singly-linked lists"},
          {"source": "<source title>", "page": 112, "kind": "figure"}]}
```

- `type` ∈ `text | diagram | multihop | cross | unanswerable`.
- `quote`: verbatim, ≤ 40 words (shorter than the fixed chunker's 64-token overlap, so it lies wholly in at
  least one fixed chunk).
- Unanswerable rows have `gold: []`.
- The plan's `gold_chunk_ids` is replaced by `gold` because chunk ids differ per ingest/corpus; matched chunk
  ids are recorded in results instead.
- The 3 placeholder EXAMPLE rows are deleted.
- Starter target: ~8 text, 4 diagram, 3 multihop, 5 unanswerable.

### 5. Metrics — `eval/metrics.py` (pure functions)

- `normalise(text)`: lowercase, join hyphenated line breaks, collapse whitespace.
- `matches(hit, gold_item)`: same source title, and either `normalise(quote) in normalise(hit.text)`, or
  (figure gold) `hit.kind == "figure" and hit.page == gold.page`.
- `recall_at(hits, gold, k=5)`: share of gold items matched by any of the top-k hits.
- `mrr(hits, gold)`: 1 / rank of the first hit matching any gold item, else 0.
- `citation_accuracy(answer, hits, gold)`: share of `[n]` markers in the answer whose hit matches gold
  (None if the answer cites nothing).
- `cohen_kappa(a, b)`: plain Python.
- Unanswerable rows are excluded from retrieval metrics; `refused` rate is reported on them only.

### 6. Runner — `python -m eval.run --config <name> [--limit N] [--dataset PATH]`

Replaces the flag-based runner. Applies the config's overrides to `settings`, opens its corpus, then per row:

- **Non-agent:** `hits = get_retriever().retrieve(question, topic_id)` (k=6), then the same generation step
  as `core.query._answer`. `_answer` is split into retrieve + `_generate(question, hits)` so the eval
  reuses the prompt without retrieving twice; `_answer`'s behaviour is unchanged.
- **Agent (H):** `open_graph()` on a throwaway agent db in the corpus dir, fresh `thread_id` per question; records the final
  answer and the documents in final state.
- **Per row records:** `id, type, question, hits[{source_title, page, chunk_id, kind, score, text}], answer,
  refused, latency_s, prompt_tokens, completion_tokens, cost_usd, error`.
- `refused` = answer starts with `This isn't in your material` (shared by `ANSWER_SYSTEM` and the agent's
  `not_found`).

Output: `eval/results/<config>.json` = `{"config": {name, overrides}, "run_at", "git_sha", "rows": [...]}`.
Metrics are not stored by the runner; they are recomputed from hits at report time.

### 7. Judge — `python -m eval.judge <config>` and `--calibrate`

Uses `llm.complete_json` with `model=settings.judge_model or settings.llm_model`.

- **Faithfulness:** numbered context + answer → `{"claims": [{"claim", "supported"}]}`; score = supported /
  total. Skipped for refusals.
- **Correctness:** question + gold_answer + answer → `{"verdict": "correct|partial|wrong"}` → 1 / 0.5 / 0.
  Skipped for unanswerable rows (refusal covers those).
- Scores are written into each row as `judge: {faithfulness, correctness}`.
- Unparseable reply: one retry, then `null`, counted as "judge failures" in the report (never coerced).
- Cache: `eval/results/judge_cache.json`, key = sha256(judge model, prompt version, question, answer,
  context). Re-judging is free unless one of those changes.
- Bias guards: the judge never sees the config name; the answer is always in the same prompt slot.
- **`--calibrate`:** samples 30 judged rows across configs (stratified by type and by judge verdict). For each it shows
  question, gold answer, answer and context, and asks you for correctness (c/p/w) and "all claims supported?"
  (y/n). Labels are appended to `eval/judge_labels.jsonl` (committed). Then it prints % agreement and κ for both
  (faithfulness made yes/no: score == 1). Already-labelled rows are skipped, so the session can be resumed.

### 8. Report — `python -m eval.report`

Loads every `eval/results/*.json`, computes metrics, and replaces the README section between
`<!-- eval:start -->` and `<!-- eval:end -->` (appended if missing; the rest of the README is untouched) with:

1. Dataset size by type; judge agreement line (n, % agree, κ per metric; "judge unvalidated" if κ < 0.6 or
   no labels).
2. Main table: Config · Change · Recall@5 · MRR · Faithfulness · Correctness · Refusal on unanswerable ·
   Latency (s) · Cost/q ($). Row A: "not measured (Cognee removed in Phase 1)". Missing configs are omitted.
3. Diagram-only sub-table (Recall@5, Correctness) for F vs G.
4. One Mermaid `xychart-beta` bar chart: Recall@5 by config.

### 9. Labeller — `python -m eval.label`

Runs against the structure corpus.

- `propose --n N --type text|diagram`: samples chunks spread over sources and pages (figure chunks for
  `diagram`) not yet used in the dataset. qwen drafts `{question, gold_answer, quote}` as JSON. For text, the
  quote must be verbatim in the chunk (else re-ask once, else skip). You accept / edit (each field) / skip /
  quit. Accepted rows are appended with the next free id.
- `find "<question>" --type multihop|unanswerable|text|cross`: shows the top 10 hits; you enter the numbers of
  the gold hits (empty = unanswerable). Quotes are auto-taken from each ticked chunk (its first sentence ≤ 40 words that
  is verbatim; figure hits become figure gold), then you type the gold answer.

## Testing (offline, existing conftest fakes, TDD)

- metrics: normalisation, quote and figure matching, Recall@5, MRR, citation accuracy, κ.
- retrieval switches: `rerank=False` skips the reranker (fake asserts it is not called) and keeps RRF order;
  `figures=False` drops figure hits.
- llm usage tally increments per call.
- runner: planted corpus → `--config` writes the expected JSON shape; refused flag for the non-agent path and the
  agent path (fake LLM).
- judge: parsing, retry then null, cache hit avoids an LLM call; κ/agreement from fixed labels.
- report: fixed result files → README marker section replaced, surrounding text byte-identical; row A text;
  chart block present.
- labeller: verbatim-quote check, accept/skip/edit with scripted stdin, id allocation.

## Order of work (one commit per step)

1. metrics.py
2. rerank / figures switches + judge_model setting + llm usage tally
3. configs.py + new runner (+ `_answer` split)
4. judge + calibrate
5. report
6. labeller
7. eval.ingest; fetch and ingest *Open Data Structures* (check the licence; check that figures extract)
8. **You:** label ~20 questions
9. Run all configs (pick best F, set G, run G, H)
10. **You:** calibrate the judge (30 labels)
11. Report → README; commit results

## Out of scope

CI smoke eval (Phase 7), the tutoring-simulation metric, ColPali (row I), running row A, cross-topic
questions in the starter set, multi-judge voting, matplotlib.

## Risks

- Steps 8 and 10 need you; real numbers cannot land without them (~1 h total).
- A 7B judge grading 7B answers tends to be lenient; κ on n=30 has wide error, and the README states n.
- LLM-proposed single-chunk questions are easy and inflate Recall@5; hand-written multihop and unanswerable
  rows counter that.
- A quote crossing a structure-chunker split fails to match; the labeller's verbatim check only covers the
  structure corpus.
- Captioning the corpus twice with the 7B vision model is slow (once).
