from __future__ import annotations
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
)

class QdrantStore:

    def __init__(self, dim: int | None = None, collection: str = "daily_planet", url: str | None = None, recreate: bool = False):
        self.client = QdrantClient(
            url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        self.collection = collection
        exists = self.client.collection_exists(collection)
        if recreate and exists:
            self.client.delete_collection(collection)
            exists = False

        if not exists:
            if dim is None:
                from rag.embed import embed_query
                dim = len(embed_query("dimension_probe"))
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def add(self, vectors, chunks):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                payload=chunk,
            )
            for vec, chunk in zip(vectors, chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, k=3, section=None):
        query_filter = None
        if section:
            variations = list({section, section.lower(), section.capitalize(), section.title(), section.upper()})
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="metadata.section",
                        match=MatchAny(any=variations),
                    )
                ]
            )

        q_vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=q_vec,
                limit=k,
                query_filter=query_filter,
            )
            return [(hit.score, hit.payload) for hit in response.points]
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=q_vec,
                limit=k,
                query_filter=query_filter,
            )
            return [(hit.score, hit.payload) for hit in hits]