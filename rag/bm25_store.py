from __future__ import annotations
import re
from rank_bm25 import BM25Okapi

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25Store:
    def __init__(self, chunks):
        self.chunks=chunks
        tokenized_chunks=[_tokenize(c["text"]) for c in chunks]
        self.bm25=BM25Okapi(tokenized_chunks)

    def search(self, query, k=3):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in ranked]
