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
    assert list(body) == ["llm", "vision"] and body["llm"]["ok"]
    # The mocked Ollama lists only qwen2.5:7b
    assert body["vision"]["ok"] is False and "ollama pull qwen2.5vl:7b" in body["vision"]["error"]


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
        "file": ("notes.pdf", _pdf_bytes("Linear algebra basics", "Gradient descent minimises the loss"), "application/pdf"),  # noqa: E501
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
    async def fake_connect(a, b):
        return {"summary": f"{a}~{b}"}

    monkeypatch.setattr(agent.query, "connect_topics", fake_connect)
    ev = _events(client.post("/agent", json={"message": "connect cells with tissues", "thread_id": "y"}))
    assert ev[-1]["event"] == "done" and ev[-1]["answer"] == "cells~tissues"


def test_agent_history_unknown_thread_is_empty(client):
    assert client.get("/agent/nope").json() == {"messages": []}


def test_figure_endpoint(client, tmp_path):
    from core.parse import Block

    sid = topics.add_source("general", "pdf", "slides")["id"]
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNG fake")
    retrieval.index_chunks(sid, "general", [Block(1, None, "Figure (p.1): x")], figure={"image_path": str(img), "bbox": "0,0,1,1"})  # noqa: E501
    retrieval.index_chunks(sid, "general", [Block(1, None, "| t |", kind="table")])
    with session() as s:
        fig_id, table_id = s.exec(select(Chunk.id).order_by(Chunk.id)).all()
    resp = client.get(f"/figures/{fig_id}")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png" and resp.content == b"\x89PNG fake"
    assert client.get(f"/figures/{table_id}").status_code == 404
    assert client.get("/figures/9999").status_code == 404
    img.unlink()
    assert client.get(f"/figures/{fig_id}").status_code == 404


def test_learner_endpoint_orders_mastery_and_counts_due(client):
    from datetime import date, timedelta
    from core.db import ConceptMastery, LearnerFact
    from tests.planting import plant

    plant(["a", "b", "c", "d"], {i: [(n, "is", n)] for i, n in enumerate(["alpha", "beta", "gamma", "delta"])})
    today, tomorrow = date.today().isoformat(), (date.today() + timedelta(1)).isoformat()
    with session() as s:
        s.add_all([ConceptMastery(concept="gamma", ease=2.6, next_review=tomorrow, last_verdict="correct", attempts=1),
                   ConceptMastery(concept="beta", ease=2.36, next_review=tomorrow, last_verdict="partial", attempts=1, wrong=1),  # noqa: E501
                   ConceptMastery(concept="alpha", ease=1.96, next_review=tomorrow, last_verdict="wrong", attempts=1, wrong=1),  # noqa: E501
                   ConceptMastery(concept="delta", ease=2.6, next_review=today, last_verdict="correct", attempts=2),
                   # its source was deleted: review can't ask it, so the panel must not list it
                   ConceptMastery(concept="orphan", ease=1.3, next_review=today, last_verdict="wrong", attempts=3, wrong=3),  # noqa: E501
                   LearnerFact(fact="Exam on 12 Dec")])
        s.commit()
    body = client.get("/learner").json()
    assert [m["concept"] for m in body["mastery"]] == ["delta", "alpha", "beta", "gamma"]
    # the review rule: due today, or last answered wrong / partly right
    assert body["to_review"] == 3 and [f["fact"] for f in body["facts"]] == ["Exam on 12 Dec"]
    assert set(body["mastery"][0]) >= {"concept", "ease", "interval_days", "next_review", "last_verdict", "attempts", "wrong"}  # noqa: E501


def test_delete_fact(client):
    from core.db import LearnerFact

    with session() as s:
        s.add(LearnerFact(fact="Prefers diagrams"))
        s.commit()
    fid = client.get("/learner").json()["facts"][0]["id"]
    assert client.delete(f"/learner/facts/{fid}").status_code == 200
    assert client.delete(f"/learner/facts/{fid}").status_code == 404
    assert client.get("/learner").json()["facts"] == []


def test_audio_does_not_speak_a_failure_message(client, monkeypatch):
    spoken = []
    monkeypatch.setattr(topics, "synthesize_audio", lambda tid, script: spoken.append(script))
    client.post("/ingest/text", json={"text": "Self-attention weighs tokens", "title": "L1", "topic_id": "general"})

    async def down(*a, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(topics.llm, "complete", down)
    assert client.post("/topics/general/audio").status_code == 503 and spoken == []
