from __future__ import annotations

from typing import List, Optional
import json
try:
    from app.models import SearchResult
except ImportError:
    from models import SearchResult

from rag.embed import embed_query
from rag.qdrant_store import QdrantStore
from rag.vector_store import FaissStore
from rag.bm25_store import BM25Store
from rag.answer import answer_question

_store: Optional[QdrantStore] = None

_bm25_store = None
_faiss_store = None


def _get_stores() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def _get_stores():
    global _bm25_store, _faiss_store
    if _bm25_store is None:
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        _bm25_store = BM25Store(chunks)
        _faiss_store = FaissStore.load()

    return _bm25_store, _faiss_store

def search(query: str, k: int = 3, section: Optional[str] = None) -> List[SearchResult]:
    store = _get_store()
    query_vector = embed_query(query)
    raw_results = store.search(query_vector=query_vector, k=k, section=section)

    results: List[SearchResult] = []
    for score, payload in raw_results:
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        results.append(
            {
                "score": float(score),
                "source": metadata.get("source", ""),
                "section": metadata.get("section", ""),
                "text": payload["text"][:200] if payload.get("text") else "",
            }
        )
    return results

def ask(question: str):
    """Grouned, cited or honestly-declined answers"""
    bm25_store, faiss_store = _get_stores()
    return answer_question(question, bm25_store, faiss_store)