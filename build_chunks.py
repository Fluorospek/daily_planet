import json
from rag.ingest import ingest_folder
from rag.chunk import chunk_documents, count_tokens

def main():

    docs = ingest_folder("data")
    print(f"Ingested {len(docs)} documents")

    for doc in docs:
        print(f" - {doc.metadata.get('source', '?'):32}"
            f"{count_tokens(doc.content):>5} tokens"
            f"[{doc.metadata.get('section','?')}]")

    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    print(f"\nSplit into {len(chunks)} chunks")

    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk.content, "metadata": chunk.metadata}) + "\n")

    print("Wrote chunks.jsonl")

if __name__ == "__main__":
    main()