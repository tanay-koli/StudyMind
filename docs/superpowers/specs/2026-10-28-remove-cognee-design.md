# Remove Cognee (design)

Decision (2026-10-28): drop Cognee Cloud entirely. It was never configured on this machine, so every feature still using it already failed. The eval baseline becomes a local dense-only retriever. This changes the master plan's "Cognee baseline" item.

## Feature rewiring (our retriever + `llm.complete`)

Every feature sends a short retrieval query and a separate generation prompt; never one string for both.

| Feature | Retrieval | Generation |
|---|---|---|
| `cross_topic_ask(question, primary, all_ids)` | `question`: primary topic k=6, each other topic with chunks k=2, concurrently | one call; two-section format `**From <name>:**` / `**Connected from other topics:**`; `[n]` citations over all hits; split into `primary_section` / `cross_section` as today |
| `summarize_topic(topic: str)` | `topic`, all topics, k=8 | "Summarize what the material says about '<topic>'" with `[n]` citations |
| `connect_topics(a, b)` | `a` and `b`, all topics, k=6 each | A: `build_connection_graph` over the hit texts; B (if A empty): LLM lists `CONCEPT: x \| REASON: y` lines from both contexts, parsed by `_parse_concept_lines`; C: empty state |
| `generate_study_guide`, `generate_audio_script` | topic name, scoped to the topic, k=8 | the existing structure prompts over the numbered context |

Responses gain `citations` (same shape as `/ask`) where they generate text: cross-topic, summarize, study guide/audio are cached strings and do not.
"No material" checks use the topic's source count instead of Cognee datasets.

## Removals

`settings.cognee_base_url/cognee_api_key`, `_headers`, `check_cognee_cloud`, the `cognee` entry in `/health`; `query._recall`, `_extract_text_from_recall`, `_extract_sources_from_recall`; `topics._llm_call`, `_extract_text`, `get_topic_datasets`; ingest's `_upload_to_cloud`, dataset-name helpers and `result["cognee"]`/`result["dataset_name"]`; `graph_utils.parse_search_results`, `format_source_attribution`; `query._build_text_summary`; `CogneeRetriever`; Cognee strings in UI and README.

`Source.dataset_name` is removed from the model; `init_db()` drops the column once when present (`ALTER TABLE source DROP COLUMN dataset_name`). `/topics` loses the `datasets` key.

## Baseline

`DenseRetriever`: dense top-k only (no BM25, RRF or rerank), selected by `RETRIEVER=dense`; `eval.run --retriever hybrid|dense`.

## Tests

The `offline` fixture only allows Ollama `/api/tags`. Each rewired feature has a test asserting it retrieves with the short query (not the instruction) and generates from the retrieved chunks. Existing tests drop Cognee expectations.
