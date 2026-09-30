"""Full answering pipeline: retrieve -> rerank -> check confidence -> generate
or refuse
"""

from __future__ import annotations
from rag.embed import embed_query
from rag.hybrid import hybrid_search
from rag.reranker import rerank
from rag.generate import generate_answer

CONFIDENCE_THRESHOLD = 0.0

def answer_question(question, bm25_store, faiss_store, k=4, candidate_k=20):
    """Retrieve, rerank, and either generate a grounded answer or refuse"""
    query_vector = embed_query(question)
    candidates = hybrid_search(
        query_vector=query_vector,
        query=question,
        k=candidate_k,
        candidate_k=candidate_k,
        bm25_store=bm25_store,
        faiss_store=faiss_store
    )

    reranked = rerank(question, candidates, top_k=k)

    top_score = reranked[0][0] if reranked else float("-inf")
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I can't verify that from the Daily Planet's sources.",
            "sources": [],
            "confidence": "low"
        }
    
    chunks = [chunk for _score, chunk in reranked]
    answer_text = generate_answer(question, chunks)
    sources = [
        {"source": c['metadata']['source'], "chunk_id": c['metadata']['chunk_id']} for c in chunks
    ]
    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": "high"
    }