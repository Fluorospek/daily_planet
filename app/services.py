from __future__ import annotations

from typing import List, Optional

try:
    from app.models import SearchResult
except ImportError:
    from models import SearchResult

from rag.embed import embed_query
from rag.qdrant_store import QdrantStore

_store: Optional[QdrantStore] = None


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


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