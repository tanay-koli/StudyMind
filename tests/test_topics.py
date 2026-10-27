"""StudyMind — topic/source storage (SQLite) and cached generation."""

import json

import pytest

from core import db, topics


def test_general_always_exists_and_cannot_be_deleted():
    assert topics.GENERAL_TOPIC_ID in topics.load_topics()
    assert topics.delete_topic(topics.GENERAL_TOPIC_ID) is False


def test_create_topic_dedupes_by_name():
    a = topics.create_topic("Machine Learning")
    b = topics.create_topic("  machine learning ")
    assert a["id"] == b["id"] == "machine_learning"
    assert len(topics.load_topics()) == 2  # General + ML


def test_sources_drive_datasets_and_invalidate_caches():
    tid = topics.create_topic("Physics")["id"]
    topics._update_topic(tid, study_guide="old guide")

    src = topics.add_source(tid, tid, "note", "Newton")
    topic = topics.load_topics()[tid]
    assert topic["datasets"] == [tid]
    assert topic["study_guide"] is None  # new material clears the cache

    topics.delete_source(src["id"])
    assert topics.load_topics()[tid]["datasets"] == []


def test_move_source_between_topics():
    a = topics.create_topic("A")["id"]
    b = topics.create_topic("B")["id"]
    src = topics.add_source(a, a, "note", "n")
    assert topics.move_source(src["id"], b)["topic_id"] == b
    assert topics.list_sources(a) == []
    assert topics.move_source(src["id"], "missing") is None


def test_delete_topic_removes_its_sources():
    tid = topics.create_topic("Chemistry")["id"]
    topics.add_source(tid, tid, "pdf", "Atoms")
    assert topics.delete_topic(tid) is True
    assert topics.list_sources() == []


def test_delete_all_sources():
    topics.add_source(topics.GENERAL_TOPIC_ID, "note_a", "note", "a")
    topics.add_source(topics.GENERAL_TOPIC_ID, "note_b", "note", "b")
    assert topics.delete_all_sources() == 2
    assert topics.list_sources() == []


def test_legacy_topics_json_is_imported_once(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "general": {"id": "general", "name": "General", "created_at": "2026-01-01T00:00:00",
                    "color": "#A4B0BE", "datasets": ["note_test_note"]},
        "bio": {"id": "bio", "name": "Biology", "created_at": "2026-02-01T00:00:00",
                "color": "#7C5CFC", "datasets": ["bio"], "study_guide": "cached"},
    }), encoding="utf-8")
    monkeypatch.setattr(topics, "LEGACY_TOPICS_FILE", legacy)
    monkeypatch.setattr(db.settings, "database_url", f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    db.reset_engine()

    topics.init_db()
    topics.init_db()  # second start must not duplicate anything

    loaded = topics.load_topics()
    assert set(loaded) == {"general", "bio"}
    assert loaded["bio"]["study_guide"] == "cached"
    assert loaded["general"]["datasets"] == ["note_test_note"]
    assert [s["source_type"] for s in topics.list_sources()] == ["note", "unknown"]


@pytest.mark.asyncio
async def test_study_guide_is_cached(offline):
    tid = topics.create_topic("Transformers")["id"]
    topics.add_source(tid, tid, "note", "n")

    first = await topics.generate_study_guide(tid)
    recalls = offline.count("/api/v1/recall")
    second = await topics.generate_study_guide(tid)

    assert first == second
    assert "Self-attention" in first
    assert offline.count("/api/v1/recall") == recalls  # served from the DB


@pytest.mark.asyncio
async def test_suggest_topic_name_uses_llm():
    assert await topics.suggest_topic_name("Attention heads...", []) == "Transformer Models"
