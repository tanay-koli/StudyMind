"""
StudyMind — Chunk store (embedded Qdrant) and retrievers.

Each chunk is a SQLite row and a Qdrant point with the same id: a dense vector
("dense") and a BM25 sparse vector ("bm25"), payload {source_id, topic_id}.
One collection per embedding model, so models can be compared.
"""

import asyncio
import threading
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

from qdrant_client import QdrantClient, models
from sqlmodel import func, select

from core.config import DATA_DIR, settings
from core.db import Chunk, Source, session
from core.parse import Block

# name -> (fastembed model, dim, query prefix, document prefix)
EMBED_MODELS: dict[str, tuple[str, int, str, str]] = {
    "bge-small": ("BAAI/bge-small-en-v1.5", 384, "", ""),
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", 768, "search_query: ", "search_document: "),
}
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "BAAI/bge-reranker-base"
COLLECTION_PREFIX = "chunks_"
_BATCH = 64
# Not fastembed's default %TEMP%, which Windows cleanup wipes (~1.2 GB to re-download)
MODEL_CACHE = str(DATA_DIR / "models")
CANDIDATES = 30  # per retriever, and fused candidates sent to the reranker
RRF_K = 60


# ---------------------------------------------------------------------------
# Models (fastembed, loaded once per process; tests replace these functions)
# ---------------------------------------------------------------------------

# lru_cache does not lock: concurrent first calls (cross-topic fans out per topic)
# would each load, and on a fresh install download, the same model
_model_lock = threading.Lock()


@lru_cache(maxsize=None)
def _load_dense(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=MODEL_CACHE)


@lru_cache(maxsize=1)
def _load_sparse():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(SPARSE_MODEL, cache_dir=MODEL_CACHE)


@lru_cache(maxsize=1)
def _load_rerank():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL, cache_dir=MODEL_CACHE)


def _dense_model(name: str):
    with _model_lock:
        return _load_dense(name)


def _sparse_model():
    with _model_lock:
        return _load_sparse()


def _rerank_model():
    with _model_lock:
        return _load_rerank()


def embed_dense(texts: list[str], *, query: bool) -> list[list[float]]:
    name, _, q_prefix, d_prefix = EMBED_MODELS[settings.embed_model]
    prefix = q_prefix if query else d_prefix
    return [v.tolist() for v in _dense_model(name).embed([prefix + t for t in texts])]


def embed_sparse(texts: list[str], *, query: bool) -> list[models.SparseVector]:
    model = _sparse_model()
    embs = model.query_embed(texts) if query else model.embed(texts)
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embs]


def rerank(query: str, texts: list[str]) -> list[float]:
    return list(_rerank_model().rerank(query, texts))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None
# Embedded Qdrant has no internal locking, and ingest, delete and ask call it from
# worker threads; every Qdrant call goes through this lock (embedding stays outside).
_lock = threading.RLock()


def store() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            if settings.qdrant_path == ":memory:":
                _client = QdrantClient(location=":memory:")
            else:
                _client = QdrantClient(path=settings.qdrant_path)
        return _client


def reset_store() -> None:
    """Drop the cached client so the next call uses settings.qdrant_path again."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = None


def collection_name() -> str:
    return COLLECTION_PREFIX + settings.embed_model.replace("-", "_")


def _chunk_collections() -> list[str]:
    with _lock:
        return [c.name for c in store().get_collections().collections if c.name.startswith(COLLECTION_PREFIX)]


def _ensure_collection() -> str:
    name = collection_name()
    client = store()
    with _lock:
        if client.collection_exists(name):
            return name
        dim = EMBED_MODELS[settings.embed_model][1]
        client.create_collection(
            name,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # No-ops in embedded mode (it warns); they matter once Qdrant runs as a server
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            client.create_payload_index(name, "source_id", models.PayloadSchemaType.INTEGER)
            client.create_payload_index(name, "topic_id", models.PayloadSchemaType.KEYWORD)
    return name


def _search_text(c: Chunk) -> str:
    """What gets embedded and reranked: the heading often names the topic the body never repeats."""
    return f"{c.section_heading}\n{c.text}" if c.section_heading else c.text


def _upsert(chunks: list[Chunk]) -> None:
    name = _ensure_collection()
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i:i + _BATCH]
        texts = [_search_text(c) for c in batch]
        dense = embed_dense(texts, query=False)
        sparse = embed_sparse(texts, query=False)
        points = [
            models.PointStruct(
                id=c.id,
                vector={"dense": d, "bm25": sp},
                payload={"source_id": c.source_id, "topic_id": c.topic_id},
            )
            for c, d, sp in zip(batch, dense, sparse)
        ]
        with _lock:
            store().upsert(name, points)


def index_chunks(source_id: int, topic_id: str, blocks: list[Block]) -> int:
    """Write Chunk rows, then their Qdrant points. Returns the chunk count."""
    rows = [
        Chunk(source_id=source_id, topic_id=topic_id, chunker=settings.chunker, ord=i,
              page=b.page, section_heading=b.section_heading, text=b.text)
        for i, b in enumerate(blocks)
    ]
    if not rows:
        return 0
    with session() as s:
        s.add_all(rows)
        s.commit()
        for r in rows:
            s.refresh(r)
    _upsert(rows)
    return len(rows)


def reindex() -> int:
    """Embed every Chunk row into the current model's collection."""
    with session() as s:
        rows = s.exec(select(Chunk)).all()
    if rows:
        _upsert(list(rows))
    return len(rows)


def ensure_indexed() -> None:
    """Reindex when the current model's collection is missing chunks (e.g. after switching EMBED_MODEL)."""
    with session() as s:
        rows = s.exec(select(func.count(Chunk.id))).one()
    name = collection_name()
    with _lock:
        points = store().count(name).count if store().collection_exists(name) else 0
        if points > rows:
            # Upserts never remove orphan points (e.g. the DB was reset, Qdrant kept): rebuild
            store().delete_collection(name)
    if points != rows:
        reindex()


def _source_filter(source_ids: list[int]) -> models.FilterSelector:
    return models.FilterSelector(filter=models.Filter(must=[
        models.FieldCondition(key="source_id", match=models.MatchAny(any=source_ids)),
    ]))


def delete_source_points(source_ids: list[int]) -> None:
    """Remove the sources' points from every model's collection."""
    if not source_ids:
        return
    with _lock:
        for name in _chunk_collections():
            store().delete(name, points_selector=_source_filter(source_ids))


def move_source_points(source_id: int, topic_id: str) -> None:
    with _lock:
        for name in _chunk_collections():
            store().set_payload(name, payload={"topic_id": topic_id}, points=_source_filter([source_id]))


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    chunk_id: Optional[int]
    source_id: Optional[int]
    source_title: str
    page: Optional[int]
    section: Optional[str]
    text: str
    score: float


class Retriever(Protocol):
    async def retrieve(
        self, query: str, topic_id: Optional[str], k: int = 6, rerank_query: Optional[str] = None,
    ) -> list[Hit]: ...


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over rankings, rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _topic_filter(topic_id: Optional[str]) -> Optional[models.Filter]:
    if not topic_id:
        return None
    return models.Filter(must=[models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))])


def _hydrate(ids: list[int]) -> list[tuple[Chunk, str]]:
    """Chunk rows with their source title; orphan points (no Source row) drop out."""
    with session() as s:
        return s.exec(
            select(Chunk, Source.title).join(Source, Chunk.source_id == Source.id).where(Chunk.id.in_(ids))
        ).all()


def _hit(c: Chunk, title: str, score: float) -> Hit:
    return Hit(c.id, c.source_id, title, c.page, c.section_heading, c.text, float(score))


class HybridRetriever:
    """Dense + BM25, fused with RRF, then reranked with a cross-encoder."""

    async def retrieve(self, query: str, topic_id: Optional[str], k: int = 6,
                       rerank_query: Optional[str] = None) -> list[Hit]:
        return await asyncio.to_thread(self._search, query, topic_id, k, rerank_query)

    def _search(self, query: str, topic_id: Optional[str], k: int,
                rerank_query: Optional[str] = None) -> list[Hit]:
        name = collection_name()
        client = store()
        with _lock:
            if not client.collection_exists(name):
                return []
        flt = _topic_filter(topic_id)

        dense = embed_dense([query], query=True)[0]
        sparse = embed_sparse([query], query=True)[0]
        with _lock:
            rankings = [
                [p.id for p in client.query_points(
                    name, query=vector, using=using, query_filter=flt, limit=CANDIDATES,
                ).points]
                for vector, using in ((dense, "dense"), (sparse, "bm25"))
            ]
        fused = rrf(rankings)[:CANDIDATES]
        rows = _hydrate(fused) if fused else []
        if not rows:
            return []
        scores = rerank(rerank_query or query, [_search_text(c) for c, _ in rows])
        ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[:k]
        return [_hit(c, title, score) for (c, title), score in ranked]


class DenseRetriever:
    """Baseline for the eval: dense vectors only, no BM25, fusion or rerank."""

    async def retrieve(self, query: str, topic_id: Optional[str], k: int = 6,
                       rerank_query: Optional[str] = None) -> list[Hit]:
        return await asyncio.to_thread(self._search, query, topic_id, k, rerank_query)

    def _search(self, query: str, topic_id: Optional[str], k: int,
                rerank_query: Optional[str] = None) -> list[Hit]:
        name = collection_name()
        client = store()
        vector = embed_dense([query], query=True)[0]
        with _lock:
            if not client.collection_exists(name):
                return []
            points = client.query_points(
                name, query=vector, using="dense", query_filter=_topic_filter(topic_id), limit=k,
            ).points
        scores = {p.id: p.score for p in points}
        rows = _hydrate(list(scores)) if scores else []
        rows.sort(key=lambda r: scores[r[0].id], reverse=True)
        return [_hit(c, title, scores[c.id]) for c, title in rows]


def base_retriever() -> Retriever:
    return DenseRetriever() if settings.retriever == "dense" else HybridRetriever()


def get_retriever(rewrite: bool = True) -> Retriever:
    base = base_retriever()
    if not rewrite or settings.query_rewrite == "none":
        return base
    from core.rewrite import RewritingRetriever  # rewrite imports this module

    return RewritingRetriever(base, settings.query_rewrite)
