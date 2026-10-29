# Phase 2 — Query understanding and grounded generation: design

Date: 2026-10-28. Builds on Phase 1 (merged at afa7cca).

## Goal

Every generation feature is grounded and structured. Done when:

1. a study guide covers every section of a topic, and
2. the topic map shows real shared concepts.

## Already done in Phase 1 (not in scope)

- Separate retrieval query and generation prompt for every feature.
- Inline citation per claim and "This isn't in your material." for ask / ask_scoped.
- Per-topic retrieval and single synthesis for cross-topic.

## Constraints

- Everything local: Qwen 2.5 7B via Ollama on an RTX 3060, fastembed on CPU.
- Tests stay offline with fakes (fake `llm.complete`, fake embedders, `:memory:` Qdrant).
- Storage changes are new tables only (`create_all()`, no migrations).
- Write files as UTF-8 explicitly.

## Cross-cutting fix: Ollama context window

Nothing sets `num_ctx`, so Ollama uses its runtime default (2048 or 4096 tokens,
depending on version) and silently truncates longer prompts. `_grounded` already
sends about 4k tokens. `LLMClient.complete` passes `num_ctx=8192` for `ollama/*`
models unless the caller overrides it.

## 1. Query rewriting

New module `core/rewrite.py` with `RewritingRetriever`, which wraps the retriever
from `get_retriever()` and keeps the `retrieve(query, topic_id, k)` interface, so
every feature and `eval/run.py` use it unchanged.

- Setting `QUERY_REWRITE`: `none | multi | hyde`, default `none` (baseline
  unchanged until the eval shows a gain). `eval/run.py` gets `--rewrite`.
- `multi`: one LLM call returns 3 paraphrases. Retrieve the original plus the 3
  paraphrases, fuse the 4 rankings with the existing `rrf()`, return the top k.
- `hyde`: one LLM call writes a short hypothetical answer, which becomes the
  retrieval query. The reranker still scores against the original question; the
  retrievers accept `rerank_query=` for this (defaults to `query`; `DenseRetriever`
  does not rerank and ignores it).
- If the rewrite call fails or returns nothing usable, retrieve with the original
  query. Rewriting never breaks retrieval.

## 2. Cross-topic decomposition

`cross_topic_ask` changes:

1. One LLM call decomposes the question into at most 3 sub-questions. On failure,
   or a single sub-question, use the original question.
2. Retrieve each sub-question per topic, skipping topics with no chunks.
   De-duplicate by `chunk_id`. Budgets are unchanged: 6 primary, 2 per other topic.
3. Synthesis stays the single existing generation call over those scoped chunks
   only. The prompt, response keys and UI are unchanged.

## 3. Study guide map-reduce

New module `core/studyguide.py`.

**Units.** Load every `Chunk` of the topic ordered by `(source_id, ord)`. Group
consecutive chunks by `(source_id, section_heading)`. Pack adjacent groups of the
same source into one unit while it stays within 2,500 tokens (counted with the bge
tokenizer via `token_spans`). A section over budget is split into windows. Chunks
with no heading form plain windows. Each unit records source title, page range and
its headings.

**Map.** One LLM call per unit, run in sequence, `format="json"`, validated with
Pydantic: `{"key_points": [str], "terms": [{"term": str, "definition": str}]}`.
Invalid output gets one retry. If it still fails, the section stays in the guide
with key point "(could not summarise this section)" and the guide is not cached.

**Reduce.** One LLM call over the map outputs' key points and terms only (never raw
chunks) produces `overview`, `key_concepts` and `review_questions`. The `sections`
list is copied from the map outputs, not generated, so every section is always
present.

**Model.**

```python
class Term(BaseModel):
    term: str
    definition: str

class SectionSummary(BaseModel):
    heading: str
    source_title: str
    pages: str
    key_points: list[str]

class StudyGuide(BaseModel):
    topic: str
    overview: str
    sections: list[SectionSummary]
    key_concepts: list[str]
    terms: list[Term]
    review_questions: list[str]
```

**Cache and API.** `topic.study_guide` stores the `StudyGuide` JSON. A cached value
that does not parse as a `StudyGuide` (the old Markdown) is stale and regenerated.
`POST /topics/{id}/study-guide` returns `{"guide": <Markdown rendered from the
model>, "data": <StudyGuide or null>}`. The UI keeps rendering `guide`; its request
timeout for this endpoint rises to 900 s.

**Out of scope.** The audio script stays on `_grounded`.

## 4. Concept extraction and topic connections

**Tables** (new):

```python
class ConceptEdge(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(index=True)
    source_id: int = Field(index=True)
    topic_id: str = Field(index=True)
    head: str
    relation: str
    tail: str
    head_norm: str = Field(index=True)
    tail_norm: str = Field(index=True)

class ConceptExtraction(SQLModel, table=True):
    # A chunk has been processed, even if it yielded no triples
    chunk_id: int = Field(primary_key=True)
```

**Worker** (`core/concepts.py`). One asyncio task started in the API lifespan:

- Loop: select up to 4 chunks without a `ConceptExtraction` row, make one LLM call
  (`format="json"`), write their edges and done rows in one transaction. Repeat
  until nothing is pending, then wait on an `asyncio.Event`.
- `ingest._store` sets the event after indexing. Startup runs the loop, which
  backfills existing chunks.
- Prompt: chunks numbered 1-4, at most 5 triples per chunk, concepts as short noun
  phrases. Reply: `{"chunks": [{"n": 1, "triples": [["head", "relation", "tail"]]}]}`.
  Triples with an empty part, a part over 60 characters, or head equal to tail are
  dropped.
- A failed batch stays pending, is logged, and is retried on the next wake or
  restart (`shortcut:` comment — a permanently failing chunk retries every restart).
- On write, chunk ids that no longer exist are skipped (source deleted mid-call).

**Normalisation** `normalise(name)`: lowercase, strip punctuation, collapse
whitespace, drop a leading "the"/"a"/"an", strip a trailing "s" unless the word
ends in "ss", "is" or "us". `shortcut:` no lemmatisation or synonyms; upgrade if
topic maps look fragmented.

**Delete and move.** `_purge` deletes `ConceptEdge` and `ConceptExtraction` rows of
the purged chunks. `move_source` updates `ConceptEdge.topic_id`.

**`connect_topics(topic_a, topic_b)`.** No LLM call.

1. Retrieve each free-text topic as today, with k = 15.
2. Collect normalised concepts (head and tail) from the edges of each side's chunks.
3. Shared concepts = intersection, minus the two topic names (normalised).
4. Rank by the total number of distinct chunks mentioning the concept across both
   sides (ties: alphabetical); keep the top 8.

The graph keeps its shape (topic A — concept — topic B), with edges labelled by a
real relation from a supporting triple. The summary lists each concept with one
supporting triple per side and its citation (title · p.N). If any retrieved chunk
is still pending, the summary adds "N of M passages are still being processed — try
again shortly." No shared concepts gives the existing empty state.
`connection_method` becomes `"concept_overlap"`. `_connect_via_llm` and
`_parse_concept_lines` are removed with their tests.

**Known cost.** Ollama serves one request at a time; during a backfill a question
can wait up to one batch call (about 20-40 s). The worker does not yield to user
queries.

## Error handling summary

| Failure | Behaviour |
|---|---|
| Rewrite / decompose LLM call fails | Fall back to the original query |
| Study guide map unit fails twice | Section marked, guide returned, not cached |
| Study guide reduce fails | Guide returned without overview etc., not cached |
| Concept batch fails | Stays pending, retried later |
| Chunk deleted during extraction | Its triples are discarded |

## Testing

All offline with the fake LLM returning canned JSON or text.

- Rewrite: multi fuses all 4 rankings; hyde retrieves with the hypothetical text
  and reranks with the original; both fall back when the LLM raises.
- Cross-topic: decomposition retrieves only the given topics and skips empty ones;
  fallback on LLM failure.
- Study guide: packing respects the budget and splits oversized sections;
  headingless windows; every heading appears in `sections` even when the reduce
  reply omits it; bad-JSON retry; failed unit means not cached; legacy Markdown
  cache is regenerated.
- Concepts: worker stores edges and done rows; failed batch stays pending; deleted
  chunk skipped; delete purges and move updates edges; `normalise` cases;
  `connect_topics` overlap from planted edges excludes topic names; pending note.
- `llm.complete` passes `num_ctx` for ollama models.

## Deferred

- Priority for user queries over background extraction.
- Lemmatisation / synonym merging of concepts.
- Background job for study guide generation.
