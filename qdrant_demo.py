import json
from rag.embed import embed_texts, embed_query
from rag.qdrant_store import QdrantStore


def main():
    with open("chunks.jsonl", "r") as f:
        chunks = [json.loads(line) for line in f]
    vectors = embed_texts([c["text"] for c in chunks])
    dim = len(vectors[0])
    store = QdrantStore(dim=dim)
    store.add(vectors, chunks)
    print(f"Added {len(chunks)} chunks in qdrant")

    query="How will the city pay for the new sports venue?"
    query_vector = embed_query(query)

    print(f"Query: {query}")

    for score, chunk in store.search(query_vector, k=3):
        meta = chunk["metadata"]
        print(f"    [{score:.3f}]  {meta.get('source')}  ({meta.get('section')})")

    print("\n Same query, but filtered for section 'weather' only:")
    for score, chunk in store.search(query_vector, k=3,section="weather"):
        meta = chunk["metadata"]
        print(f"    [{score:.3f}]  {meta.get('source')}  ({meta.get('section')})")

if __name__ == "__main__":
    main()
