"""The answer flow now with end-to-end langfuse tracing, observing the retrieval, reranking and generation
"""

import json

from langfuse import get_client, observe

from rag.embed import embed_query
from rag.bm25_store import BM25Store
from rag.generate import generate_answer
from rag.hybrid import hybrid_search
from rag.judge import judge_faithfulness
from rag.reranker import rerank
from rag.vector_store import FaissStore

CONFIDENCE_THRESHOLD = 0.0

@observe(as_type="retriever")
def traced_retrieve(question, bm25_store, faiss_store, candidate_k=20):
    query_vector = embed_query(question)
    return hybrid_search(
        query=question,
        bm25_store=bm25_store,
        faiss_store=faiss_store,
        query_vector=query_vector,
        k=candidate_k,
        candidate_k=candidate_k
    )

@observe(as_type="span")
def traced_rerank(question, candidates, top_k=4):
    return rerank(query=question, candidates=candidates, top_k=top_k)

@observe(as_type="generation")
def traced_generate(question, chunks):
    return generate_answer(question, chunks)

@observe(name="answer_question")
def traced_answer_question(question, bm25_store, faiss_store):
    """Same answering logic as previously implemented but wrapped in @observe to create a span 
    in the trace "answer_question" for evaluation purposes where each stage - "retriever", "reranker",
    "generation" - is itself wrapped in @observe to create nested spans.
    """
    candidates = traced_retrieve(question, bm25_store, faiss_store)
    reranked = traced_rerank(question, candidates)

    top_score = reranked[0][0] if reranked else float("-inf")
    client = get_client()
    trace_id = client.get_current_trace_id()

    if top_score < CONFIDENCE_THRESHOLD:
        client.update_current_span(output={"confidence": "low"})
        return {
            "answer": "I can't verify that from the Daily Planet's sources.",
            "sources": [],
            "confidence": "low",
            "trace_id": trace_id
        }

    chunks = [chunks for _score, chunks in reranked]
    answer_text = traced_generate(question, chunks)

    sources = [
        {"source": c['metadata'].get('source', ''),
        "section": c['metadata'].get('section', ''),
        "chunk_id": c['metadata'].get('chunk_id', ''),
        "text": c['text']}
        for c in chunks
    ]

    sources_text = "\n\n".join(f"[{s["source"]}]\n{s['text']}" for s in sources)
    faithful, reason = judge_faithfulness(answer_text, sources_text)
    client.score_current_trace(name="faithfulness", value=1.0 if faithful else 0.0, comment=reason)

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": "high",
        "trace_id": trace_id
    }

def main():
    with open("chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    
    bm25_store = BM25Store(chunks)
    faiss_store = FaissStore.load()
    
    question = "How is the city protecting taxpayers from cost overruns on the arena?"
    result = traced_answer_question(question, bm25_store, faiss_store)

    print(f"Query: {question}")
    print(f"Confidence: {result.get("confidence")}")
    print(f"\nAnswer:\n{result.get('answer')}")
    print(f"\nTrace ID: {result.get('trace_id')}")

    client = get_client()
    client.flush()
    print(f"\nView this trace: {client.get_trace_url(trace_id=result['trace_id'])}")

if __name__ == "__main__":
    main()
