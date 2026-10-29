# 🧠 StudyMind — Personal Study Assistant

**StudyMind** is a personal study assistant that runs entirely on your machine. It indexes your study material with its own retrieval pipeline (hybrid search + reranking) and answers with a local LLM, citing the source and page for every claim.

> 🖥️ **Local and free, with NotebookLM features** — no cloud account or API key needed. Organizes your material into Notebooks (Topics), generates Study Guides, synthesizes Audio Overview podcasts, and discovers cross-topic connections.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📚 Topics (Notebooks)** | Group your ingested resources into collections (e.g. "ML Course", "DSA Prep") with custom color tags. |
| **📥 Multi-Source Ingest** | Upload PDFs, YouTube transcripts, web articles, or paste your own notes. PDFs keep page numbers and section headings. |
| **💡 AI Topic Suggestions** | Suggests which topic new material belongs to via a toast/banner confirmation. |
| **🤖 Study Agent** | One chat that routes each message (answer, explain, connect topics), rewrites and retries weak searches, and checks its answer against the cited passages. You watch each step as it happens. |
| **💬 Scoped Q&A with citations** | Ask within one topic or across all; answers cite passages as `[n]` and show clickable chips like `📄 notes.pdf · p.12`. |
| **🔍 Cross-Topic Connections** | Answer from one topic and surface related passages from the others. |
| **📖 Auto Study Guides** | Map-reduce over every section of a topic: per-section key points, terms, overview and review questions. Exportable to Markdown (`.md`). |
| **🎙️ Audio Overviews** | A podcast-style script, spoken locally (`.wav`) via `pyttsx3`. |
| **🔗 Topic Map Visualization** | Discover how topics and concepts relate through interactive node graphs. |

---

## 🏗️ Architecture

```
┌──────────────────────┐   HTTP   ┌──────────────────────────────────────────┐
│ Streamlit UI         │ ───────▶ │ FastAPI service (api/main.py)            │
│ ui/app.py            │          │  topics · ingest · sources · ask ·       │
│ (thin client)        │          │  summarize · connect · study guide/audio │
└──────────────────────┘          └───────┬───────────┬───────────┬──────────┘
                                          │           │           │
                              ┌───────────▼──┐ ┌──────▼──────┐ ┌──▼────────────────┐
                              │ SQLite       │ │ Qdrant      │ │ LLMClient         │
                              │ (SQLModel)   │ │ (embedded)  │ │ (LiteLLM → Ollama)│
                              │ topic,source,│ │ dense + BM25│ └─────┬─────────────┘
                              │ chunk        │ │ vectors     │       │ traces (optional)
                              └──────────────┘ └─────────────┘       ▼
                                                                  Langfuse
```

**Retrieval:** PyMuPDF blocks → structure-aware chunks (512 tokens) → bge-small dense vectors + BM25 sparse vectors in Qdrant → Reciprocal Rank Fusion (k = 60) → bge-reranker-base picks the top 6. Every feature sends a short retrieval query (the question or topic) and a separate generation prompt.

---

## 📁 Project Structure

```
StudyMind/
├── .env.example           # Optional settings — copy to .env to change defaults
├── requirements.txt       # Python dependencies (new ones pinned)
├── api/
│   └── main.py            # FastAPI app: every feature as an HTTP endpoint
├── core/
│   ├── config.py          # pydantic-settings, shared async HTTP client, LLM health check
│   ├── db.py              # SQLModel tables: topic, source, chunk
│   ├── llm.py             # LLMClient (LiteLLM): token/cost/latency logging, Langfuse
│   ├── parse.py           # PDF → blocks with page and section heading
│   ├── chunking.py        # Fixed (512/64) and structure-aware chunkers
│   ├── retrieval.py       # Qdrant store, HybridRetriever, DenseRetriever (baseline)
│   ├── ingest.py          # Parse, chunk and index each source
│   ├── query.py           # Grounded Q&A, cross-topic, summarize, connect
│   ├── rewrite.py         # Query rewriting (multi-query, HyDE) around the retriever
│   ├── concepts.py        # Concept triples (background worker) and topic overlap
│   ├── agent.py           # LangGraph agent: routing, retry on weak search, self-check
│   ├── studyguide.py      # Map-reduce study guide over every section
│   ├── topics.py          # Topic + source CRUD, suggestion, study guide, audio overview
│   └── graph_utils.py     # Build connection graphs, render pyvis HTML
├── eval/
│   ├── dataset.jsonl      # Eval questions (add ~15 a week)
│   └── run.py             # python -m eval.run --retriever hybrid|dense
├── ui/
│   └── app.py             # Streamlit UI — calls the API only
├── data/                  # studymind.db, agent.db, qdrant/, models/, uploads/, audio/ (git-ignored)
└── tests/                 # Offline: Ollama, embedders and reranker are faked
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** for the local LLM (or set `LLM_MODEL` to a hosted model)
- About **1.2 GB** of disk for the embedding, BM25 and reranker models (downloaded to `data/models/` on first use)

### Step 1: Clone & Install Dependencies

```bash
git clone https://github.com/tanay-koli/StudyMind.git
cd StudyMind
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

### Step 2: Configure (optional)

Everything has a default. To change it, create a `.env`:

```env
LLM_MODEL=ollama/qwen2.5:7b        # or ollama/gemma3:4b, groq/..., anthropic/...
EMBED_MODEL=bge-small              # bge-small | nomic (reindexes on next start)
CHUNKER=structure                  # structure | fixed
RETRIEVER=hybrid                   # hybrid | dense (baseline)
QUERY_REWRITE=none                 # none | multi | hyde (extra LLM call per question)
RELEVANCE_MIN=-3.0                 # reranker score a passage needs to count as relevant (hybrid only)
AGENT_DB_PATH=data/agent.db        # where agent conversations are kept
```

> Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to trace every LLM call in Langfuse.

### Step 3: Run StudyMind

```bash
uvicorn api.main:app --reload      # API on http://localhost:8000 (docs at /docs)
streamlit run ui/app.py            # UI on http://localhost:8501 (second terminal)
```

The sidebar shows **🟢 Connected** for the LLM when Ollama is running and the model is pulled.

Sources added before local indexing existed have no chunks: re-add them to make them searchable.
The first API start after upgrading backfills concepts with the LLM in the background (roughly one LLM call per 4 chunks); a question can wait up to one batch call while it runs, and the topic map shows "N of M passages are still being processed" until it finishes.

---

## 🧪 Running Tests

```bash
pytest -v
```

All tests run offline: Ollama is mocked with `httpx.MockTransport`, the LLM, embedders and reranker are faked, Qdrant runs in memory, and each test gets a temporary SQLite database.

## 📏 Evaluation

```bash
python -m eval.run --retriever hybrid
python -m eval.run --retriever dense    # baseline: dense vectors only, no BM25/RRF/rerank
python -m eval.run --retriever hybrid --rewrite multi   # query rewriting: none | multi | hyde
```

Results go to `eval/results/<retriever>-<embed>-<date>.jsonl`. Stop the API first: embedded Qdrant allows one process at a time.

---

## 📖 Usage Guide

### 1. Create a Topic (Notebook)
In the sidebar, expand **➕ New Topic**, enter a name (e.g. `"Transformers Course"`), and click **Create Topic**. It is assigned a color bullet and saved in `data/studymind.db`.

### 2. Ingest Material
Go to the **📥 Ingest** tab, select your topic dropdown, and add content:
- **PDF** — Upload a file.
- **YouTube** — Paste a video link (transcript is fetched).
- **Web Article** — Paste a link.
- **Notes** — Paste raw typed notes.

The content is chunked and indexed locally; a banner then suggests a topic for it.

### 3. Chat with the Agent
In the **🤖 Agent** tab, just ask. The sidebar topic scopes the search.
- A step list shows what it is doing: searching → checking sources → (rewriting query → searching) → answering → verifying.
- If the passages it finds score too low, it rewrites the search (at most twice), then says honestly that the material does not cover it.
- If its answer contains a sentence the passages do not support, it rewrites the answer once, then shows a warning naming the sentence.
- Pick two topics under **Connect with…**, or ask "connect X with Y", for a topic map. "Quiz me" and "review" arrive in Phase 5.
- Conversations are kept in `data/agent.db`; **🆕 New conversation** starts a fresh one.

### 4. Ask Questions
In the **💬 Ask** tab:
- Choose your scope: **All Topics** or a specific topic notebook.
- Click a citation chip under an answer to read the exact passage.
- Toggle **🔍 Find connections across topics** for a two-part answer: your topic first, then related points from the others.

### 5. Topic Study Guides & Audio Overviews
In the sidebar next to each topic row:
- Click `[📖]` (Study Guide) to generate key concepts and review questions. Download the output directly as `.md`.
- Click `[🎙️]` (Audio Overview) to generate a conversational script and play the voice-over inside Streamlit.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| API | [FastAPI](https://fastapi.tiangolo.com) |
| Storage | SQLite via [SQLModel](https://sqlmodel.tiangolo.com) |
| Vector store | [Qdrant](https://qdrant.tech) (embedded mode) |
| Embeddings, BM25, reranker | [fastembed](https://github.com/qdrant/fastembed): bge-small-en-v1.5, Qdrant/bm25, bge-reranker-base |
| LLM | [LiteLLM](https://docs.litellm.ai) → [Ollama](https://ollama.com) (Qwen 2.5 by default) |
| Tracing | [Langfuse](https://langfuse.com) (optional) |
| Speech Engine (TTS) | [pyttsx3](https://pyttsx3.readthedocs.io/) |
| PDF extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Web scraping | [trafilatura](https://trafilatura.readthedocs.io/) |
| YouTube transcripts | [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) |
| Graph visualization | [pyvis](https://pyvis.readthedocs.io/) + [networkx](https://networkx.org/) |
| HTTP client | [httpx](https://www.python-httpx.org/) |

---

## 📝 License

Built for hackathon purposes.
