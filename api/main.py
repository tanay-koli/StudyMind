"""
StudyMind API — every feature the UI uses, as HTTP endpoints.

Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from core import agent, concepts, ingest, query, retrieval, topics
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
    concepts.start()
    async with agent.open_graph() as graph:
        app.state.agent = graph
        yield
    await concepts.stop()
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


class AgentIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(min_length=1, max_length=100)
    topic_id: Optional[str] = None
    intent: Optional[Literal["qa", "explain", "connect", "quiz", "review"]] = None
    topics: Optional[list[str]] = Field(default=None, max_length=2)


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
    # Unique name so same-named uploads don't clash; deleted after ingest so deleting the source forgets it
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    save_path.write_bytes(await file.read())
    try:
        return await ingest.ingest_pdf(str(save_path), title=title or name, topic_id=topic_id)
    finally:
        save_path.unlink(missing_ok=True)


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@app.post("/agent")
async def agent_turn(body: AgentIn):
    """One chat turn, streamed as NDJSON: a line per node, then done (or error)."""
    graph = app.state.agent
    cfg = {"configurable": {"thread_id": body.thread_id}}
    turn = {"messages": [], "question": body.message, "topic_id": body.topic_id,
            "topics": body.topics or [], "intent_hint": body.intent}

    async def lines():
        try:
            async for update in graph.astream(turn, cfg, stream_mode="updates"):
                for node, out in update.items():
                    event = {"event": "node", "node": node, "label": agent.LABELS.get(node, node)}
                    if node == "retrieve":
                        event["query"] = out["queries_tried"][-1]
                    elif node == "rewrite_query":
                        event["query"] = out["query"]
                    yield json.dumps(event) + "\n"
            st = (await graph.aget_state(cfg)).values
            yield json.dumps({"event": "done", **{k: st.get(k) for k in
                              ("intent", "answer", "citations", "warning", "graph")}}) + "\n"
        except Exception as exc:
            yield json.dumps({"event": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/agent/{thread_id}")
async def agent_history(thread_id: str):
    st = await app.state.agent.aget_state({"configurable": {"thread_id": thread_id}})
    return {"messages": st.values.get("messages", [])}
