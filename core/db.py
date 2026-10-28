"""
StudyMind — SQLite storage via SQLModel.

Tables for later phases (chunk, quiz_attempt, concept_mastery) are added when
something writes them; create_all() picks up new tables without a migration.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from core.config import settings


class Topic(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    color: str
    study_guide: Optional[str] = None
    study_guide_generated_at: Optional[str] = None
    audio_script: Optional[str] = None
    audio_generated_at: Optional[str] = None


class Source(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    topic_id: str = Field(foreign_key="topic.id", index=True)
    # Cognee datasetName the text was uploaded to
    dataset_name: str
    source_type: str
    title: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Chunk(SQLModel, table=True):
    # Also the Qdrant point id
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    topic_id: str = Field(index=True)
    chunker: str
    ord: int
    page: Optional[int] = None
    section_heading: Optional[str] = None
    text: str


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine so the next call uses settings.database_url again."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def session() -> Session:
    return Session(get_engine(), expire_on_commit=False)
