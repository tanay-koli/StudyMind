"""
StudyMind API — every feature the UI uses, as HTTP endpoints.

Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import ingest, query, retrieval, topics
from core.config import DATA_DIR, check_llm, http_client, set_http_client
from core.llm import setup_tracing

UPLOAD_DIR = DATA_DIR / "uploads"
FORGET_NOTE = "Source and its indexed chunks deleted."


@asynccontextmanager
async def lifespan(app: FastAPI):
    topics.init_db()
    setup_tracing()
    # Catch up after an EMBED_MODEL switch; first run with a new model embeds every chunk
    await asyncio.to_thread(retrieval.ensure_indexed)
    yield
    await http_client().aclose()
    set_http_client(None)
    retrieval.reset_store()  # release the embedded Qdrant file lock


app = FastAPI(title="StudyMind API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class TopicIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class SuggestIn(BaseModel):
    excerpt: str = Field(min_length=1, max_length=5000)


class UrlIn(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    topic_id: Optional[str] = None


class TextIn(BaseModel):
    text: str = Field(max_length=500_000)
    title: str = Field(default="Untitled Note", max_length=200)
    topic_id: Optional[str] = None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    topic_id: Optional[str] = None
    mode: Literal["global", "scoped", "cross"] = "global"


class ConnectIn(BaseModel):
    topic_a: str = Field(min_length=1, max_length=200)
    topic_b: str = Field(min_length=1, max_length=200)


class SummarizeIn(BaseModel):
    topic: str = Field(min_length=1, max_length=200)


class MoveIn(BaseModel):
    topic_id: str


def _require_topic(topic_id: Optional[str]) -> dict:
    topic = topics.load_topics().get(topic_id or topics.GENERAL_TOPIC_ID)
    if topic is None:
        raise HTTPException(404, f"Topic '{topic_id}' not found")
    return topic


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"llm": await check_llm()}


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

@app.get("/topics")
def list_topics():
    return topics.load_topics()


@app.post("/topics")
def create_topic(body: TopicIn):
    return topics.create_topic(body.name)


@app.delete("/topics/{topic_id}")
def delete_topic(topic_id: str):
    if not topics.delete_topic(topic_id):
        raise HTTPException(400, "Topic not found or cannot be deleted")
    return {"status": "success"}


@app.post("/topics/suggest")
async def suggest_topic(body: SuggestIn):
    names = [t["name"] for t in topics.load_topics().values()]
    return {"name": await topics.suggest_topic_name(body.excerpt, names)}


@app.post("/topics/{topic_id}/study-guide")
async def study_guide(topic_id: str, force: bool = False):
    _require_topic(topic_id)
    return await topics.generate_study_guide(topic_id, force=force)


@app.post("/topics/{topic_id}/audio-script")
async def audio_script(topic_id: str, force: bool = False):
    _require_topic(topic_id)
    return {"script": await topics.generate_audio_script(topic_id, force=force)}


@app.post("/topics/{topic_id}/audio")
async def audio(topic_id: str):
    _require_topic(topic_id)
    script = await topics.generate_audio_script(topic_id)
    path = await asyncio.to_thread(topics.synthesize_audio, topic_id, script)
    if path is None:
        raise HTTPException(503, "Audio synthesis unavailable (is pyttsx3 installed?)")
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    topic_id: Optional[str] = Form(None),
):
    _require_topic(topic_id)
    # Keep only the base name so a crafted filename can't escape UPLOAD_DIR
    name = Path(file.filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / name
    save_path.write_bytes(await file.read())
    return await ingest.ingest_pdf(str(save_path), title=title or name, topic_id=topic_id)


@app.post("/ingest/youtube")
async def ingest_youtube(body: UrlIn):
    _require_topic(body.topic_id)
    return await ingest.ingest_youtube(body.url, topic_id=body.topic_id)


@app.post("/ingest/url")
async def ingest_url(body: UrlIn):
    _require_topic(body.topic_id)
    return await ingest.ingest_url(body.url, topic_id=body.topic_id)


@app.post("/ingest/text")
async def ingest_text(body: TextIn):
    _require_topic(body.topic_id)
    return await ingest.ingest_text(body.text, title=body.title, topic_id=body.topic_id)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@app.get("/sources")
def list_sources(topic_id: Optional[str] = None):
    return topics.list_sources(topic_id)


@app.patch("/sources/{source_id}")
def move_source(source_id: int, body: MoveIn):
    src = topics.move_source(source_id, body.topic_id)
    if src is None:
        raise HTTPException(404, "Source or topic not found")
    return src


@app.delete("/sources/{source_id}")
def delete_source(source_id: int):
    src = topics.delete_source(source_id)
    if src is None:
        raise HTTPException(404, "Source not found")
    return {"status": "success", "source": src, "note": FORGET_NOTE}


@app.delete("/sources")
def delete_all_sources():
    return {"status": "success", "deleted": topics.delete_all_sources(), "note": FORGET_NOTE}


# ---------------------------------------------------------------------------
# Q&A, summary, connections
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask(body: AskIn):
    if body.mode == "global":
        return await query.ask(body.question)
    if not body.topic_id:
        raise HTTPException(400, f"mode '{body.mode}' needs a topic_id")
    _require_topic(body.topic_id)
    if body.mode == "scoped":
        return await query.ask_scoped(body.question, body.topic_id)
    return await query.cross_topic_ask(body.question, body.topic_id, list(topics.load_topics()))


@app.post("/summarize")
async def summarize(body: SummarizeIn):
    return await query.summarize_topic(body.topic)


@app.post("/connect")
async def connect(body: ConnectIn):
    return await query.connect_topics(body.topic_a, body.topic_b)
