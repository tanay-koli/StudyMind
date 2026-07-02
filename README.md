# 🧠 StudyMind — Personal Study Assistant

**StudyMind** is a personal study assistant powered by [Cognee Cloud](https://app.cognee.ai) — an AI memory platform that builds a knowledge graph from your study material and lets you query it intelligently.

> ☁️ **Cloud-powered** — No local LLM, no Docker, no heavy hardware required. Ingestion and reasoning run on Cognee Cloud; only text extraction happens locally.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📥 Multi-Source Ingestion** | Ingest PDFs, YouTube transcripts, web articles, and typed notes |
| **💬 Q&A Across Sources** | Ask questions and get answers synthesized from ALL your material |
| **🔗 Connect Topics** | ⭐ Hero feature — discover how two topics relate through your knowledge graph |
| **📋 Topic Summaries** | Get cross-source summaries powered by Cognee Cloud's graph reasoning |
| **📚 Knowledge Base** | Track ingested sources and manage your session |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                   │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Ingest   │  │   Ask (Q&A)  │  │  Connect Topics    │  │
│  │  Tab      │  │   Tab        │  │  Tab (Hero)        │  │
│  └────┬─────┘  └──────┬───────┘  └────────┬───────────┘  │
│       │               │                   │               │
├───────┼───────────────┼───────────────────┼───────────────┤
│       ▼               ▼                   ▼               │
│  ┌─────────┐   ┌───────────┐      ┌─────────────┐        │
│  │ingest.py│   │ query.py  │      │graph_utils.py│        │
│  └────┬────┘   └─────┬─────┘      └──────┬──────┘        │
│       │              │                    │               │
│  Local text          │                    │               │
│  extraction          │                    │               │
│  (PyMuPDF,           │                    │               │
│  trafilatura,        │                    │               │
│  yt-transcript)      │                    │               │
├───────┼──────────────┼────────────────────┼───────────────┤
│       ▼              ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Cognee Cloud REST API                   │  │
│  │  POST /api/v1/remember  ←  ingest text as file      │  │
│  │  POST /api/v1/recall    ←  GRAPH_COMPLETION / CHUNKS │  │
│  │  GET  /api/v1/datasets/ ←  health check             │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
studymind/
├── .env.example           # Template — copy to .env and add your credentials
├── .env                   # Your Cognee Cloud credentials (git-ignored, never committed)
├── .gitignore
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── app.py                 # Streamlit UI — single-page app with 3 tabs
├── core/
│   ├── __init__.py
│   ├── config.py          # Loads .env, pings Cognee Cloud health endpoint
│   ├── ingest.py          # Extracts text locally, uploads to Cognee Cloud
│   ├── query.py           # Calls /api/v1/recall for Q&A, topics, summaries
│   └── graph_utils.py     # Parse recall results, build graphs, render pyvis HTML
├── data/                  # Uploaded files (git-ignored except .gitkeep)
└── tests/
    ├── __init__.py
    └── test_ingest.py     # Unit + integration smoke tests
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+**
- A **Cognee Cloud** account with an API key — sign up at [app.cognee.ai](https://app.cognee.ai)

### Step 1: Clone & Install Dependencies

```bash
git clone https://github.com/tanay-koli/StudyMind.git
cd StudyMind
pip install -r requirements.txt
```

### Step 2: Configure Cognee Cloud Credentials

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
COGNEE_BASE_URL=https://tenant-YOUR_TENANT_ID.aws.cognee.ai
COGNEE_API_KEY=your_api_key_here
```

> 💡 Get your `COGNEE_BASE_URL` and `COGNEE_API_KEY` from the [Cognee Cloud console](https://app.cognee.ai) under **Integrations → API / MCP → Step 1**.

### Step 3: Run StudyMind

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The sidebar will show **🟢 Connected** if your credentials are valid.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Pure unit tests (dataset naming, graph utilities) run without any credentials. Integration tests that call Cognee Cloud will be skipped automatically if `COGNEE_API_KEY` is not set.

---

## 📖 Usage Guide

### 1. Ingest Material

Go to the **📥 Ingest** tab and add your study material:

- **PDF** — Upload a PDF file (text is extracted locally, then uploaded to the cloud)
- **YouTube** — Paste a YouTube URL (transcript is fetched locally, then uploaded)
- **Web Article** — Paste an article URL (text is scraped locally, then uploaded)
- **Notes** — Type or paste your own notes

> ⚡ Ingestion is fast — text is uploaded to Cognee Cloud in seconds. Knowledge graph building happens server-side.

### 2. Ask Questions

Go to the **💬 Ask** tab and type any question. StudyMind will:
- Query the Cognee Cloud knowledge graph (`GRAPH_COMPLETION`)
- Return an answer synthesized from all your sources
- Show which sources contributed to the answer

### 3. Connect Topics (⭐ Hero Feature)

Go to the **🔗 Connect Topics** tab:
1. Enter two topic names (e.g., `"transformers"` and `"attention mechanism"`)
2. Click **Find Connection**
3. See an **interactive graph** showing how the topics relate through your knowledge graph
4. Read the **connection summary** below the graph

### 4. Manage Sources

The **sidebar** shows all ingested sources for this session. You can:
- See the source type, title, and timestamp
- Click 🗑️ to remove a source from the session view
- Click **Forget Everything** to clear the session

---

## ⚙️ Configuration

Edit `.env` to customize:

| Variable | Description |
|----------|-------------|
| `COGNEE_BASE_URL` | Your Cognee Cloud tenant URL (from the console) |
| `COGNEE_API_KEY` | Your Cognee Cloud API key (from the console) |
| `LOG_LEVEL` | Logging verbosity (`INFO`, `DEBUG`, `WARNING`) |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Sidebar shows 🔴 Disconnected | Check `COGNEE_BASE_URL` and `COGNEE_API_KEY` in `.env` |
| `401 Unauthorized` error | API key is wrong or expired — regenerate it in the Cognee Cloud console |
| `Cannot reach Cognee Cloud` | Check your internet connection |
| Import errors | Make sure you're running from the `studymind/` directory |
| PDF has no extractable text | The PDF may be image-only (scanned). Try a text-based PDF |
| YouTube transcript unavailable | The video may have disabled transcripts or be in an unsupported language |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| Knowledge Graph | [Cognee Cloud](https://app.cognee.ai) REST API |
| PDF extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Web scraping | [trafilatura](https://trafilatura.readthedocs.io/) |
| YouTube transcripts | [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) |
| Graph visualization | [pyvis](https://pyvis.readthedocs.io/) + [networkx](https://networkx.org/) |
| HTTP client | [httpx](https://www.python-httpx.org/) |

---

## 📝 License

Built for hackathon purposes. Uses [Cognee](https://github.com/topoteretes/cognee) (Apache 2.0).
