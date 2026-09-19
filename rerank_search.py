import json
from rag.embed import embed_query
from rag.vector_store import FaissStore
from rag.bm25_store import BM25Store
from rag.hybrid import hybrid_search
from rag.reranker import rerank

def main():
    chunks = [json.loads(line) for line in open("chunks.jsonl", encoding="utf-8")]

    bm25 = BM25Store(chunks)
    faiss_store = FaissStore.load()

    query = "budget vote"
    query_vector = embed_query(query)

    candidates = hybrid_search(query, bm25, faiss_store, query_vector, k=11, candidate_k=20)
    print(f"Query: {query}")
    print("--- Hybrid rank, before re-renking")
    for i, (score, chunk) in enumerate(candidates):
        meta = chunk["metadata"]
        print(f"  #{i+1} [{score:.4f}] {meta.get('source')} chunk {meta.get('chunk_index')}")

    reranked = rerank(query, candidates, top_k=5)
    print("\n --- After re-renking ---")
    for i, (score, chunk) in enumerate(reranked):
        meta = chunk["metadata"]
        print(f"  #{i+1} [{score:.4f}] {meta.get('source')} chunk {meta.get('chunk_index')}")

if __name__ == "__main__":
    main()