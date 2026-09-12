import json
from rag.embed import embed_texts
from rag.vector_store import FaissStore

def main():
    with open("chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(chunks)} chunks. Embedding the chunks now....")

    vectors = embed_texts([c["text"] for c in chunks])
    dim=len(vectors[0])
    print(f"Got back {len(vectors)} vectors, each of {dim} dimensions")

    store = FaissStore(dim)
    store.add(vectors, chunks)
    store.save()

    print("Saved Faiss index to index.faiss")


if __name__ == "__main__":
    main()