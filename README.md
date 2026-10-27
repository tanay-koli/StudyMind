# 🧠 StudyMind — Personal Study Assistant

**StudyMind** is a personal study assistant powered by [Cognee Cloud](https://app.cognee.ai) — an AI memory platform that builds a knowledge graph from your study material and lets you query it intelligently.

> ☁️ **Cloud-powered with NotebookLM Features** — No local LLM or heavy hardware required. Organizes your study materials into Notebooks (Topics), autogenerates Study Guides, synthesizes Audio Overview podcasts, and discovers cross-topic connections.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📚 Topics (Notebooks)** | Group your ingested resources into collections (e.g. "ML Course", "DSA Prep") with custom color tags. |
| **📥 Multi-Source Ingest** | Upload PDFs, YouTube transcripts, web articles, or paste your own notes. |
| **💡 AI Topic Suggestions** | Automatically analyzes ingested text excerpts and suggests which topic folder the content belongs to via toast/banner confirmation. |
| **💬 Scoped Q&A** | Query the knowledge graph scoped strictly to a single topic, or query all. |
| **🔍 Cross-Topic Connections** | Ask a question primarily using one topic, but search for and synthesize connection paths from other topics in parallel. |
| **📖 Auto Study Guides** | Generates detailed key concepts, review details, and practice questions for a topic. Exportable to Markdown (`.md`). |
| **🎙️ Audio Overviews** | Generates a podcast overview script and synthesizes local speech (`.wav`) via `pyttsx3` so you can listen to summary audio. |
| **🔗 Topic Map Visualization** | Hero feature — discover how topics and concepts relate through interactive 2D node graphs. |

---

## 🏗️ Architecture

```
┌──────────────────────┐   HTTP   ┌──────────────────────────────────────────┐
│ Streamlit UI         │ ───────▶ │ FastAPI service (api/main.py)            │
│ ui/app.py            │          │  topics · ingest · sources · ask ·       │
│ (thin client)        │          │  summarize · connect · study guide/audio │
└──────────────────────┘          └───────┬───────────┬───────────┬──────────┘
                                          │           │           │
                              ┌───────────▼──┐ ┌──────▼─────┐ ┌───▼──────────────┐
                              │ SQLite       │ │ LLMClient  │ │ Cognee Cloud API │
                              │ (SQLModel)   │ │ (LiteLLM → │ │ /remember        │
                              │ topic,source │ │  Ollama)   │ │ /recall          │
                              └──────────────┘ └─────┬──────┘ └──────────────────┘
                                                     │ traces (optional)
                                                     ▼
                                                  Langfuse
```

---

## 📁 Project Structure

```
StudyMind/
├── .env.example           # Template — copy to .env and add your credentials
├── requirements.txt       # Python dependencies (new ones pinned)
├── api/
│   └── main.py            # FastAPI app: every feature as an HTTP endpoint
├── core/
│   ├── config.py          # pydantic-settings, shared async HTTP client, health checks
│   ├── db.py              # SQLModel tables: topic, source
│   ├── llm.py             # LLMClient (LiteLLM): token/cost/latency logging, Langfuse
│   ├── topics.py          # Topic + source CRUD, suggestion, study guide, audio overview
│   ├── ingest.py          # Extracts text locally, uploads scoped to topic datasets
│   ├── query.py           # Scoped recall, cross-topic query synthesis, global Q&A
│   └── graph_utils.py     # Parse recall results, build graphs, render pyvis HTML
├── ui/
│   └── app.py             # Streamlit UI — calls the API only
├── data/                  # studymind.db, uploads/, audio/ (git-ignored)
└── tests/                 # Offline: Cognee, Ollama and the LLM are mocked
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+**
- A **Cognee Cloud** account with an API key — sign up at [app.cognee.ai](https://app.cognee.ai)
- **[Ollama](https://ollama.com)** for the local LLM (or set `LLM_MODEL` to a hosted model)

### Step 1: Clone & Install Dependencies

```bash
git clone https://github.com/tanay-koli/StudyMind.git
cd StudyMind
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

### Step 2: Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
COGNEE_BASE_URL=https://tenant-YOUR_TENANT_ID.aws.cognee.ai
COGNEE_API_KEY=your_api_key_here
LLM_MODEL=ollama/qwen2.5:7b        # or ollama/gemma3:4b, groq/..., anthropic/...
```

> 💡 Get your `COGNEE_BASE_URL` and `COGNEE_API_KEY` from the [Cognee Cloud console](https://app.cognee.ai) under **Integrations → API / MCP → Step 1**.
> Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to trace every LLM call in Langfuse.

### Step 3: Run StudyMind

```bash
uvicorn api.main:app --reload      # API on http://localhost:8000 (docs at /docs)
streamlit run ui/app.py            # UI on http://localhost:8501 (second terminal)
```

The sidebar shows **🟢 Connected** for Cognee Cloud and the LLM when both are reachable.
On first start, an existing `data/topics.json` is imported into `data/studymind.db` (the JSON file is kept as a backup).

---

## 🧪 Running Tests

```bash
pytest -v
```

All tests run offline: Cognee Cloud and Ollama are mocked with `httpx.MockTransport`, the LLM is faked, and each test gets a temporary SQLite database.

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

After ingestion, Cognee analyzes the content, fires a toast, and shows a top banner asking if you'd like to move it to a suggested AI-derived topic.

### 3. Ask Questions
In the **💬 Ask** tab:
- Choose your scope: **All Topics** or select a specific topic notebook.
- Toggle **🔍 Find connections across topics** to perform a comparative vector search and display segmented answers.

### 4. Topic Study Guides & Audio Overviews
In the sidebar next to each topic row:
- Click `[📖]` (Study Guide) to generate key concepts and review questions. Download the output directly as `.md`.
- Click `[🎙️]` (Audio Overview) to generate a conversational speech script and play a generated podcast voice-over natively inside Streamlit.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| API | [FastAPI](https://fastapi.tiangolo.com) |
| Storage | SQLite via [SQLModel](https://sqlmodel.tiangolo.com) |
| LLM | [LiteLLM](https://docs.litellm.ai) → [Ollama](https://ollama.com) (Qwen 2.5 by default) |
| Tracing | [Langfuse](https://langfuse.com) (optional) |
| Knowledge Graph | [Cognee Cloud](https://app.cognee.ai) REST API |
| Speech Engine (TTS) | [pyttsx3](https://pyttsx3.readthedocs.io/) |
| PDF extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Web scraping | [trafilatura](https://trafilatura.readthedocs.io/) |
| YouTube transcripts | [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) |
| Graph visualization | [pyvis](https://pyvis.readthedocs.io/) + [networkx](https://networkx.org/) |
| HTTP client | [httpx](https://www.python-httpx.org/) |

---

## 📝 License

Built for hackathon purposes. Uses [Cognee](https://github.com/topoteretes/cognee) (Apache 2.0).
