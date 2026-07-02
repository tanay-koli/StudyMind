# 🧠 StudyMind — Personal Study Assistant

**StudyMind** is a personal study assistant that uses [Cognee](https://github.com/topoteretes/cognee) (an open-source AI memory platform) to build a knowledge graph from your study material and let you query it intelligently.

> **100% Local** — No cloud APIs, no Docker. Runs entirely on your machine with Ollama.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📥 Multi-Source Ingestion** | Ingest PDFs, YouTube transcripts, web articles, and typed notes |
| **💬 Q&A Across Sources** | Ask questions and get answers synthesized from ALL your material |
| **🔗 Connect Topics** | ⭐ Hero feature — discover how two topics relate through your knowledge graph |
| **📋 Topic Summaries** | Get cross-source summaries with attribution (which PDF, article, or note said what) |
| **📚 Knowledge Base** | Track ingested sources, forget individual items or everything |

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
│  ┌─────────┐   ┌───────────┐      ┌─────────────┐       │
│  │ingest.py│   │ query.py  │      │graph_utils.py│       │
│  └────┬────┘   └─────┬─────┘      └──────┬──────┘       │
│       │              │                    │               │
├───────┼──────────────┼────────────────────┼───────────────┤
│       ▼              ▼                    ▼               │
│  ┌──────────────────────────────────────────────┐        │
│  │              Cognee (Knowledge Graph)         │        │
│  │  cognee.add() → cognee.cognify() → search()  │        │
│  └───────────────────┬──────────────────────────┘        │
│                      │                                    │
├──────────────────────┼────────────────────────────────────┤
│       ┌──────────────┼──────────────┐                     │
│       ▼              ▼              ▼                     │
│  ┌─────────┐   ┌──────────┐  ┌──────────┐               │
│  │ SQLite  │   │ LanceDB  │  │  KuzuDB  │               │
│  │ (data)  │   │ (vectors)│  │ (graph)  │               │
│  └─────────┘   └──────────┘  └──────────┘               │
│                                                           │
│       ┌──────────────────────────────┐                    │
│       │    Ollama (localhost:11434)   │                    │
│       │  qwen3:8b  │ nomic-embed-text│                    │
│       └──────────────────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
studymind/
├── .env                   # Ollama/Cognee config (LLM, embedding, DB providers)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── app.py                 # Streamlit UI — single-page app with 3 tabs
├── core/
│   ├── __init__.py
│   ├── config.py          # Loads .env, validates Ollama connectivity
│   ├── ingest.py          # Async functions: ingest_pdf/youtube/url/text
│   ├── query.py           # Async functions: ask, connect_topics, summarize_topic
│   └── graph_utils.py     # Parse search results, build graphs, render pyvis HTML
├── data/                  # Uploaded/sample files
└── tests/
    ├── __init__.py
    └── test_ingest.py     # Smoke tests for ingestion + graph utilities
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running ([install guide](https://ollama.ai/download))
- **16GB RAM** (GPU not required — runs on CPU)

### Step 1: Install Ollama & Pull Models

```bash
# Install Ollama (see https://ollama.ai/download for your OS)
# Then pull the required models:
ollama pull qwen3:8b
ollama pull nomic-embed-text

# Start Ollama (if not already running):
ollama serve
```

### Step 2: Clone & Install Dependencies

```bash
cd studymind
pip install -r requirements.txt
```

### Step 3: Run StudyMind

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🧪 Running Tests

```bash
# Make sure Ollama is running first
pytest tests/ -v
```

Tests that require Ollama will be skipped automatically if it's not available.

---

## 📖 Usage Guide

### 1. Ingest Material

Go to the **📥 Ingest** tab and add your study material:

- **PDF**: Upload a PDF file
- **YouTube**: Paste a YouTube URL (transcript will be extracted)
- **Web Article**: Paste an article URL (text will be extracted)
- **Notes**: Type or paste your own notes

> ⏱️ **Note**: Knowledge graph building takes 1-3 minutes per source on CPU. This is normal for local inference.

### 2. Ask Questions

Go to the **💬 Ask** tab and type any question. StudyMind will:
- Search the knowledge graph for relevant information
- Synthesize an answer from all your sources
- Show which sources contributed to the answer

### 3. Connect Topics (⭐ Hero Feature)

Go to the **🔗 Connect Topics** tab:
1. Enter two topic names (e.g., "transformers" and "attention mechanism")
2. Click "Find Connection"
3. See an **interactive graph** showing how the topics relate through your material
4. Read the **connection summary** below the graph

### 4. Manage Sources

The **sidebar** shows all ingested sources. You can:
- See the source type, title, and timestamp
- Click 🗑️ to forget individual sources
- Click "Forget Everything" to clear all data

---

## ⚙️ Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `ollama/qwen3:8b` | LLM model for reasoning |
| `EMBEDDING_MODEL` | `ollama/nomic-embed-text` | Embedding model |
| `LLM_ENDPOINT` | `http://localhost:11434` | Ollama server URL |
| `DB_PROVIDER` | `sqlite` | Relational database |
| `VECTOR_DB_PROVIDER` | `lancedb` | Vector store |
| `GRAPH_DATABASE_PROVIDER` | `kuzu` | Graph database |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot connect to Ollama" | Run `ollama serve` in a separate terminal |
| "Missing models" | Run `ollama pull qwen3:8b && ollama pull nomic-embed-text` |
| Slow ingestion (>5 min) | Normal for CPU inference. Try shorter texts first |
| "Malformed JSON" error | Local LLMs sometimes produce invalid JSON. Try re-ingesting |
| Import errors | Make sure you're in the `studymind/` directory when running |

---

## 📝 License

Built for hackathon purposes. Uses [Cognee](https://github.com/topoteretes/cognee) (Apache 2.0).
