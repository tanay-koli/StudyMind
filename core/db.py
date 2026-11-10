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
    source_type: str
    title: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Chunk(SQLModel, table=True):
    # Also the Qdrant point id. Never reused, so old citations can't point at new content;
    # shortcut: create_all applies this to new databases only, re-create data/studymind.db to get it
    __table_args__ = {"sqlite_autoincrement": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    topic_id: str = Field(index=True)
    chunker: str
    ord: int
    page: Optional[int] = None
    section_heading: Optional[str] = None
    text: str


class ConceptEdge(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(index=True)
    source_id: int = Field(index=True)
    topic_id: str = Field(index=True)
    head: str
    relation: str
    tail: str
    head_norm: str = Field(index=True)
    tail_norm: str = Field(index=True)


class ConceptExtraction(SQLModel, table=True):
    # A chunk has been processed, even if it yielded no triples
    chunk_id: int = Field(primary_key=True)


class ChunkMedia(SQLModel, table=True):
    # Side row for chunks that are a figure, a table or a stretch of video (chunk itself is unchanged)
    chunk_id: int = Field(primary_key=True)
    kind: str  # figure | table | video
    image_path: Optional[str] = None
    bbox: Optional[str] = None  # "x0,y0,x1,y1" in PDF points
    start_s: Optional[float] = None
    url: Optional[str] = None  # video deep link at start_s


class PendingFigure(SQLModel, table=True):
    # A figure crop waiting for the vision model; becomes a Chunk + ChunkMedia once captioned.
    # AUTOINCREMENT: ids are never reused, so a row deleted mid-caption can't be mistaken for a new upload's
    __table_args__ = {"sqlite_autoincrement": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(index=True)
    topic_id: str
    page: int
    section_heading: Optional[str] = None
    bbox: str
    image_path: str
    attempts: int = 0


class QuizAttempt(SQLModel, table=True):
    # Every answer the learner gives; kept when sources are deleted (it describes the learner)
    __table_args__ = {"sqlite_autoincrement": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    topic_id: Optional[str] = Field(default=None, index=True)
    concept: str = Field(index=True)  # normalised, as in concept_edge
    chunk_id: Optional[int] = None
    kind: str  # mcq | short
    question: str
    answer: str
    verdict: Optional[str] = None  # correct | partial | wrong | None (ungraded)
    feedback: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConceptMastery(SQLModel, table=True):
    # SM-2 state per concept, shared across topics
    concept: str = Field(primary_key=True)
    ease: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    next_review: str  # ISO date
    last_verdict: Optional[str] = None
    attempts: int = 0
    wrong: int = 0  # wrong + partial answers


class LearnerFact(SQLModel, table=True):
    __table_args__ = {"sqlite_autoincrement": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    fact: str
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


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
