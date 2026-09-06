from __future__ import annotations
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.ingest import Document

ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(ENCODER.encode(text))

@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)

def chunk_documents(docs, chunk_size=400, chunk_overlap=50):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks=[]
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.content)):
            meta=dict(doc.metadata)
            meta["chunk_index"]=i
            meta["chunk_id"]=f"{meta.get('source', 'doc')}-{i}"
            chunks.append(Chunk(content=piece, metadata=meta))
    return chunks