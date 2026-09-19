import json

from rag.embed import embed_query
from rag.vector_store import FaissStore
from rag.bm25_store import BM25Store
from rag.hybrid import hybrid_search

def show(label, results):
    print(f"  {label}:")
    for score, chunk in results[:3]:
        meta=chunk['metadata']
        print(f"    [{score:0.3f}] {meta.get("source")} chunk {meta.get("chunk_index")}")
        
def main():
    chunks = [json.loads(line) for line in open("chunks.jsonl", encoding="utf-8")]
    bm25 = BM25Store(chunks)
    faiss_store = FaissStore.load()

    queries = ["24-CV-1099", "sports venue"]

    for query in queries:
        print(f"\nQuery: {query}")
        vec = embed_query(query)
        show("Keyword-only (BM25)", bm25.search(query, k=3))
        show("Vector-only (Faiss)", faiss_store.search(vec, top_k=3))
        show("Hybrid (RRF)", hybrid_search(query, bm25, faiss_store, vec))

if __name__ == "__main__":
    main()
    