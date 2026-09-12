from __future__ import annotations
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

class QdrantStore:

    def __init__(self, dim, collection="daily_planet", url=None):
        self.client = QdrantClient(
            url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        self.collection = collection
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def add(self,vectors,chunks):
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=chunks
            )
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, k=3, section=None):
        query_filter = None
        if section:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section)
                    )
                ]
            )

        hits = self.client.search(
            collection_name = self.collection,
            query_vector = query_vector,
            limit=k,
            query_filter=query_filter
        )
        return [(hit.score, hit.payload) for hit in hits]