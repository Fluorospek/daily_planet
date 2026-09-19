from __future__ import annotations
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

_MODEL_NAME=os.getenv("CROSS_ENCODER_MODEL")
_model = None

def _get_model():
    global _model
    if _model is None:
        print("Loading cross encoder model...")
        _model = CrossEncoder(_MODEL_NAME)
    return _model

def rerank(query, candidates, top_k=5):

    pairs = [(query, chunk["text"]) for _score, chunk in candidates]
    cross_scores = _get_model().predict(pairs)
    reranked = sorted(zip(cross_scores, [chunk for _score, chunk in candidates]), key=lambda x: x[0], reverse=True)
    return [(float(score), chunk) for score, chunk in reranked[:top_k]]