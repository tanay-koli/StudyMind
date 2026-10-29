"""StudyMind — end-to-end API flow through FastAPI's TestClient (offline)."""

import json

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

import api.main
from sqlmodel import select

from core import agent, config, retrieval, topics
from core.db import Chunk, session
from core.llm import llm


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api.main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config.settings, "agent_db_path", (tmp_path / "agent.db").as_posix())
    with TestClient(api.main.app) as c:
        yield c


def _pdf_bytes(*pages: str) -> bytes:
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health(client):
    body = client.get("/health").json()
    assert list(body) == ["llm"] and body["llm"]["ok"]


def test_full_flow(client, offline, monkeypatch):
    tid = client.post("/topics", json={"name": "Transformers"}).json()["id"]

    ingested = client.post("/ingest/text", json={
        "text": "Self-attention...", "title": "Lecture 1", "topic_id": tid,
    }).json()
    assert ingested["status"] == "success"

    sources = client.get("/sources").json()
    assert [(s["id"], s["topic_id"]) for s in sources] == [(ingested["source_id"], tid)]

    for body in (
        {"question": "What is attention?", "mode": "global"},
        {"question": "What is attention?", "mode": "scoped", "topic_id": tid},
        {"question": "What is attention?", "mode": "cross", "topic_id": tid},
    ):
        resp = client.post("/ask", json=body)
        assert resp.status_code == 200
        assert resp.json()["citations"][0]["source_title"] == "Lecture 1"

    async def json_llm(prompt, system=None, **kw):
        return '{"key_points": ["kp"], "terms": [], "overview": "o"}'

    with monkeypatch.context() as m:
        m.setattr(llm, "complete", json_llm)
        body = client.post(f"/topics/{tid}/study-guide").json()
    assert body["guide"].startswith("# Study Guide: Transformers") and body["data"]["topic"] == "Transformers"
    assert client.post(f"/topics/{tid}/study-guide").json() == body  # cached

    assert client.post("/topics/suggest", json={"excerpt": "attention"}).json()["name"] == "Transformer Models"

    moved = client.patch(f"/sources/{ingested['source_id']}", json={"topic_id": "general"}).json()
    assert moved["topic_id"] == "general"

    assert client.delete(f"/sources/{ingested['source_id']}").json()["status"] == "success"
    assert client.get("/sources").json() == []


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "q", "mode": "scoped"}).status_code == 400
    assert client.post("/ask", json={"question": "q", "mode": "scoped", "topic_id": "nope"}).status_code == 404
    assert client.post("/ask", json={"question": "", "mode": "global"}).status_code == 422
    assert client.post("/ingest/text", json={"text": "x", "topic_id": "nope"}).status_code == 404


def test_ingest_pdf_upload(client, tmp_path):
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("../../evil.pdf", _pdf_bytes("Gradient descent"), "application/pdf")},
    )
    body = resp.json()
    assert body["status"] == "success", body
    assert body["page_count"] == 1
    assert body["excerpt"].startswith("evil.pdf\n\nGradient descent")
    # The crafted name is reduced to its base name; the upload is not kept after ingest
    assert body["title"] == "evil.pdf"
    assert list((tmp_path / "uploads").iterdir()) == []


def test_ingest_rejects_non_pdf(client):
    resp = client.post("/ingest/pdf", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_summarize_and_connect(client):
    client.post("/ingest/text", json={"text": "Self-attention weighs tokens", "title": "L1", "topic_id": "general"})
    summary = client.post("/summarize", json={"topic": "attention"}).json()
    assert summary["sources_referenced"] == ["L1"] and summary["summary"] == "Transformer Models"
    body = client.post("/connect", json={"topic_a": "attention", "topic_b": "RNNs"}).json()
    assert "summary" in body and "nodes" in body


def test_audio_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(topics, "synthesize_audio", lambda tid, script: None)
    assert client.post("/topics/general/audio").status_code == 503


def test_delete_topic(client):
    tid = client.post("/topics", json={"name": "Temp"}).json()["id"]
    assert client.delete(f"/topics/{tid}").status_code == 200
    assert client.delete("/topics/general").status_code == 400


def test_scoped_answer_cites_pages(client):
    tid = client.post("/topics", json={"name": "ML"}).json()["id"]
    client.post("/ingest/pdf", data={"topic_id": tid}, files={
        "file": ("notes.pdf", _pdf_bytes("Linear algebra basics", "Gradient descent minimises the loss"), "application/pdf"),
    })
    body = client.post("/ask", json={"question": "gradient descent", "mode": "scoped", "topic_id": tid}).json()
    top = body["citations"][0]
    assert (top["source_title"], top["page"]) == ("notes.pdf", 2)
    assert top["chunk_id"] and "Gradient descent" in top["text"]
    assert body["sources"] == ["notes.pdf"] and body["error"] is None


def test_ask_empty_topic_no_results(client):
    tid = client.post("/topics", json={"name": "Empty"}).json()["id"]
    body = client.post("/ask", json={"question": "q", "mode": "scoped", "topic_id": tid}).json()
    assert body["citations"] == [] and "No results found" in body["answer"]


def test_ask_llm_down(client, monkeypatch):
    from core.llm import llm

    async def down(prompt, system=None, **kw):
        raise RuntimeError("ollama not running")

    monkeypatch.setattr(llm, "complete", down)
    client.post("/ingest/text", json={"text": "Self-attention", "title": "L1", "topic_id": "general"})
    resp = client.post("/ask", json={"question": "attention", "mode": "global"})
    assert resp.status_code == 200 and "RuntimeError" in resp.json()["error"]


def _ingest(client, tid, text="Photosynthesis converts light into chemical energy"):
    return client.post("/ingest/text", json={"text": text, "title": "Bio", "topic_id": tid}).json()["source_id"]


def _cited(client, tid=None):
    body = {"question": "photosynthesis light", "mode": "scoped" if tid else "global"}
    if tid:
        body["topic_id"] = tid
    return [c["source_title"] for c in client.post("/ask", json=body).json()["citations"]]


@pytest.mark.parametrize("how", ["source", "topic", "all"])
def test_deleted_source_never_retrieved(client, how):
    tid = client.post("/topics", json={"name": "Biology"}).json()["id"]
    sid = _ingest(client, tid)
    assert _cited(client) == ["Bio"]
    if how == "source":
        client.delete(f"/sources/{sid}")
    elif how == "topic":
        client.delete(f"/topics/{tid}")
    else:
        client.delete("/sources")
    assert _cited(client) == []
    # Really gone, not just hidden by the join to Source
    with session() as s:
        assert s.exec(select(Chunk)).all() == []
    assert retrieval.store().count(retrieval.collection_name()).count == 0


def test_moved_source_retrieved_only_in_new_topic(client):
    t1 = client.post("/topics", json={"name": "Biology"}).json()["id"]
    t2 = client.post("/topics", json={"name": "Chemistry"}).json()["id"]
    sid = _ingest(client, t1)
    client.patch(f"/sources/{sid}", json={"topic_id": t2})
    assert (_cited(client, t1), _cited(client, t2)) == ([], ["Bio"])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _events(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_agent_streams_nodes_then_done(client, monkeypatch):
    async def search(q, tid):
        return [{"chunk_id": 1, "source_id": 1, "source_title": "N", "page": 1,
                 "section": None, "text": "t", "score": 2.0}]

    async def complete(p, system=None, **kw):
        return "A [1]."

    monkeypatch.setattr(agent, "search_material", search)
    monkeypatch.setattr(llm, "complete", complete)
    resp = client.post("/agent", json={"message": "what is t", "thread_id": "x"})
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    ev = _events(resp)
    assert [e["node"] for e in ev if e["event"] == "node"][:2] == ["route_intent", "retrieve"]
    assert ev[1]["label"] == "searching" and ev[1]["query"] == "what is t"
    assert ev[-1]["event"] == "done" and ev[-1]["answer"] == "A [1]." and ev[-1]["citations"]
    msgs = client.get("/agent/x").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_agent_error_event_then_next_turn_works(client, monkeypatch):
    async def boom(q, tid):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(agent, "search_material", boom)
    ev = _events(client.post("/agent", json={"message": "q", "thread_id": "y"}))
    assert ev[-1] == {"event": "error", "error": "RuntimeError: ollama down"}
    ev = _events(client.post("/agent", json={"message": "quiz me", "thread_id": "y"}))
    assert ev[-1]["event"] == "done" and ev[-1]["answer"] == agent.DEFERRED


def test_agent_history_unknown_thread_is_empty(client):
    assert client.get("/agent/nope").json() == {"messages": []}
