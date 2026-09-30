"""Build Qdrant index with visibility metadata, so retrieval can be filtered by who is asking.
"""
import json

from rag.embed import embed_texts
from rag.qdrant_store import QdrantStore
from rag.access import normalize_visibility

def main():
    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    chunks = normalize_visibility(chunks)
    vectors = embed_texts([c["text"] for c in chunks])
    dim = len(vectors[0])

    store = QdrantStore(dim=dim, recreate=True)
    store.add(vectors=vectors, chunks=chunks)
    print(f"Upserted {len(chunks)} chunks into Qdrant, every one tagged with a visibility tier")

if __name__ == "__main__":
    main()
    