from __future__ import annotations
import os
import torch
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model {MODEL}...")
        _embed_model = SentenceTransformer(MODEL).to(device=DEVICE)
        print("Embed model loaded.")
    return _embed_model
    
def embed_texts(texts, model=None):
    if model is None:
        model = get_embed_model()
    return model.encode(texts)

def embed_query(text, model=None):
    if model is None:
        model = get_embed_model()
    return embed_texts([text], model=model)[0]
