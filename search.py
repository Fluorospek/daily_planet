import sys
from rag.embed import embed_query
from rag.vector_store import FaissStore

def main():

    query = " ".join(sys.argv[1:]) or "How is the new arena being paid for?"
    print(f"Searching for: {query}")
    store = FaissStore.load()

    query_vector = embed_query(query)

    for score,chunk in store.search(query_vector, top_k=3):
        meta = chunk["metadata"]
        print(f"[{score:.3f}] {meta.get('source')}   ({meta.get('section')})")
        print(f"         {chunk['text'][:100].strip()}...\n")

if __name__ == "__main__":
    main()
    