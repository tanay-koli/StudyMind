"""
StudyMind — Chunk store (embedded Qdrant) and retrievers.

Each chunk is a SQLite row and a Qdrant point with the same id: a dense vector
("dense") and a BM25 sparse vector ("bm25"), payload {source_id, topic_id}.
One collection per embedding model, so models can be compared.
"""

import warnings
from functools import lru_cache

from qdrant_client import QdrantClient, models
from sqlmodel import select

from core.config import settings
from core.db import Chunk, session
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


# ---------------------------------------------------------------------------
# Models (fastembed, loaded once per process; tests replace these functions)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _dense_model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name)


@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(SPARSE_MODEL)


@lru_cache(maxsize=1)
def _rerank_model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL)


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


def store() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_path == ":memory:":
            _client = QdrantClient(location=":memory:")
        else:
            _client = QdrantClient(path=settings.qdrant_path)
    return _client


def reset_store() -> None:
    """Drop the cached client so the next call uses settings.qdrant_path again."""
    global _client
    if _client is not None:
        _client.close()
    _client = None


def collection_name() -> str:
    return COLLECTION_PREFIX + settings.embed_model.replace("-", "_")


def _chunk_collections() -> list[str]:
    return [c.name for c in store().get_collections().collections if c.name.startswith(COLLECTION_PREFIX)]


def _ensure_collection() -> str:
    name = collection_name()
    client = store()
    if not client.collection_exists(name):
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


def _upsert(chunks: list[Chunk]) -> None:
    name = _ensure_collection()
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i:i + _BATCH]
        texts = [c.text for c in batch]
        dense = embed_dense(texts, query=False)
        sparse = embed_sparse(texts, query=False)
        store().upsert(name, [
            models.PointStruct(
                id=c.id,
                vector={"dense": d, "bm25": sp},
                payload={"source_id": c.source_id, "topic_id": c.topic_id},
            )
            for c, d, sp in zip(batch, dense, sparse)
        ])


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


def _source_filter(source_ids: list[int]) -> models.FilterSelector:
    return models.FilterSelector(filter=models.Filter(must=[
        models.FieldCondition(key="source_id", match=models.MatchAny(any=source_ids)),
    ]))


def delete_source_points(source_ids: list[int]) -> None:
    """Remove the sources' points from every model's collection."""
    if not source_ids:
        return
    for name in _chunk_collections():
        store().delete(name, points_selector=_source_filter(source_ids))


def move_source_points(source_id: int, topic_id: str) -> None:
    for name in _chunk_collections():
        store().set_payload(name, payload={"topic_id": topic_id}, points=_source_filter([source_id]))
