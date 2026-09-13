from __future__ import annotations
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = os.getenv("DEVICE", None)


class LocalSentenceTransformerEmbedder:
    """Embedding model backend using local SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading local embedding model '{self.model_name}' on {device}...")
            self._model = SentenceTransformer(self.model_name).to(device=device)
            print("Local embedding model loaded.")

    def encode(self, texts: str | list[str]) -> np.ndarray:
        self._load_model()
        arr = self._model.encode(texts)
        return np.asarray(arr, dtype="float32")


class HFInferenceEmbedder:
    """Embedding model backend using Hugging Face Inference API."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", token: str | None = None):
        # Hugging Face inference API requires repository IDs in the form 'org/model'
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model_name = model_name
        self.token = token
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            token = self.token or os.getenv("HF_TOKEN")
            if not token:
                print("Warning: HF_TOKEN is not set. Inference API calls may fail or be rate limited.")
            self._client = InferenceClient(token=token)
        return self._client

    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        client = self._get_client()

        # Handle single string input
        if isinstance(texts, str):
            res = client.feature_extraction(texts, model=self.model_name)
            arr = np.asarray(res, dtype="float32")
            if arr.ndim > 1:
                arr = arr.squeeze(0)
            return arr

        texts_list = list(texts)
        if not texts_list:
            return np.empty((0,), dtype="float32")

        # Process in batches to prevent payload or timeout errors
        embeddings = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            res = client.feature_extraction(batch, model=self.model_name)
            arr = np.asarray(res, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            embeddings.append(arr)

        return np.vstack(embeddings)


_embed_model = None


def get_embed_model(
    provider: str | None = None,
    model_name: str | None = None,
    force_new: bool = False,
):
    """
    Get or create an embedding model based on provider and model name.
    If parameters are omitted, values from .env (EMBEDDING_PROVIDER, EMBEDDING_MODEL) are used.
    """
    global _embed_model

    if _embed_model is not None and provider is None and model_name is None and not force_new:
        return _embed_model

    target_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()
    target_model = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

    if target_provider in ("local", "sentence-transformers", "sentence_transformers"):
        embedder = LocalSentenceTransformerEmbedder(model_name=target_model, device=DEVICE)
    elif target_provider in ("huggingface", "hf", "inference", "hf-inference"):
        embedder = HFInferenceEmbedder(model_name=target_model, token=HF_TOKEN)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: '{target_provider}'. "
            "Supported values are 'local' or 'huggingface'."
        )

    if provider is None and model_name is None and not force_new:
        _embed_model = embedder

    return embedder


def embed_texts(texts: list[str], model=None) -> np.ndarray:
    """
    Embed a list of text strings and return a 2D numpy array of shape (N, dim).
    """
    if model is None:
        model = get_embed_model()
    return model.encode(texts)


def embed_query(text: str, model=None) -> np.ndarray:
    """
    Embed a single query string and return a 1D numpy array of shape (dim,).
    """
    if model is None:
        model = get_embed_model()
    return embed_texts([text], model=model)[0]
