from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np

class FaissStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self,vectors,chunks):
        arr = np.array(vectors).astype('float32')
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.chunks.extend(chunks)

    def search(self,query_vector,top_k=5):
        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q,top_k)
        results=[]
        for score, i in zip(scores[0], ids[0]):
            if i<0:
                continue
            results.append((float(score), self.chunks[i]))
        return results

    def save(self,path="index.faiss"):
        faiss.write_index(self.index,path)
        Path(path+".chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, path="index.faiss"):
        store = cls(1)
        store.index = faiss.read_index(path)
        store.chunks = json.loads(Path(path+".chunks.json").read_text(encoding="utf-8"))
        return store