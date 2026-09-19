from __future__ import annotations

def reciprocal_rank_fusion(ranked_lists, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    
    fused_scores = {}
    chunk_by_id = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (_score, chunk) in enumerate(ranked_list):
            chunk_id = chunk["metadata"]["chunk_id"]
            chunk_by_id[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k+rank+1)
    
    ranked_ids = sorted(fused_scores, key = lambda cid: fused_scores[cid], reverse=True)
    return [(fused_scores[cid], chunk_by_id[cid]) for cid in ranked_ids]

def hybrid_search(query, bm25_store, faiss_store, query_vector, k=10, candidate_k=20, weights=None):
    keyword_results = bm25_store.search(query, k=candidate_k)
    semantic_results = faiss_store.search(query_vector, top_k=candidate_k)
    fused = reciprocal_rank_fusion([keyword_results, semantic_results], weights=weights)
    return fused[:k]