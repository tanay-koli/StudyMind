"""
StudyMind — Eval configs: the plan's experiment ladder, each row the previous best plus one change.

corpus picks the ingested copy (the chunker runs at ingest time); agent=True answers through
the LangGraph agent instead of the plain RAG prompt. Every other key is a settings override.
"""

from core import config, db, retrieval
from core.config import DATA_DIR, settings
from core.topics import init_db

CORPUS_ROOT = DATA_DIR / "eval"

# Edit after F-multi and F-hyde have run: the one with the higher Recall@5 (tie -> F-multi)
BEST_F = "F-multi"

_E = dict(change="+ cross-encoder reranker", corpus="structure", agent=False, retriever="hybrid",
          rerank=True, figures=False, query_rewrite="none", embed_model="bge-small")
CONFIGS: dict[str, dict] = {
    "B": {**_E, "change": "Dense only, fixed 512-token chunks", "corpus": "fixed", "retriever": "dense"},
    "C": {**_E, "change": "+ structure-aware chunking", "retriever": "dense"},
    "D": {**_E, "change": "+ BM25 hybrid with RRF", "rerank": False},
    "E": _E,
    "F-multi": {**_E, "change": "+ multi-query rewriting", "query_rewrite": "multi"},
    "F-hyde": {**_E, "change": "+ HyDE rewriting", "query_rewrite": "hyde"},
}
CONFIGS["G"] = {**CONFIGS[BEST_F], "change": "+ figure captions (multimodal)", "figures": True}
CONFIGS["H"] = {**CONFIGS["G"], "change": "+ agent grade-and-retry + self-check", "agent": True}
CONFIGS["E-nomic"] = {**_E, "change": "E with nomic-embed instead of bge-small", "embed_model": "nomic"}
ORDER = list(CONFIGS)

SETTING_KEYS = ("retriever", "rerank", "figures", "query_rewrite", "embed_model")


def open_corpus(corpus: str) -> None:
    """Point the app at data/eval/<corpus>/ (never the app's own data) with the matching chunker."""
    root = CORPUS_ROOT / corpus
    root.mkdir(parents=True, exist_ok=True)
    settings.database_url = f"sqlite:///{(root / 'studymind.db').as_posix()}"
    settings.qdrant_path = (root / "qdrant").as_posix()
    settings.agent_db_path = (root / "agent.db").as_posix()
    settings.chunker = corpus
    config.FIGURE_DIR = root / "figures"  # crops too: app source ids overlap eval ones
    db.reset_engine()
    retrieval.reset_store()
    init_db()


def apply(cfg: dict) -> None:
    for key in SETTING_KEYS:
        setattr(settings, key, cfg[key])
