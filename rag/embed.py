from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class BaseEmbeddingProvider(ABC):
    """Abstract base class for provider-agnostic embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings into vector representations."""
        pass

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string into a vector representation."""
        embeddings = self.embed_texts([text])
        return embeddings[0]


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using SentenceTransformers."""

    def __init__(self, model_name: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local embedding model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name).to(device=self.device)
        print("Local embedding model loaded successfully.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI's Embeddings API."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        import openai

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please add OPENAI_API_KEY to your .env file or environment."
            )
        self.client = openai.OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]


class HuggingFaceInferenceProvider(BaseEmbeddingProvider):
    """Embedding provider using Hugging Face Inference API."""

    def __init__(self, model_name: str | None = None, token: str | None = None):
        from huggingface_hub import InferenceClient

        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
        if not self.token:
            raise ValueError(
                "HF_TOKEN environment variable is not set. "
                "Please add HF_TOKEN to your .env file or environment."
            )
        self.client = InferenceClient(model=self.model_name, token=self.token)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self.client.feature_extraction(texts)
        if isinstance(res, np.ndarray):
            if res.ndim == 3:
                res = res.mean(axis=1)
            return res.tolist()
        elif isinstance(res, list):
            arr = np.array(res)
            if arr.ndim == 3:
                arr = arr.mean(axis=1)
            return arr.tolist()
        return res


_provider_cache = {}

def get_embed_model(
    provider: str | None = None,
    model_name: str | None = None
) -> BaseEmbeddingProvider:
    """Factory function to get or create an embedding provider instance."""
    provider_key = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).lower().strip()
    model_key = model_name or os.getenv("EMBEDDING_MODEL")

    cache_key = (provider_key, model_key)
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    if provider_key in ("local", "sentence_transformers", "sentence_transformer"):
        instance = LocalEmbeddingProvider(model_name=model_name)
    elif provider_key == "openai":
        instance = OpenAIEmbeddingProvider(model_name=model_name)
    elif provider_key in ("huggingface", "hf", "hf_inference"):
        instance = HuggingFaceInferenceProvider(model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{provider_key}'. "
            "Supported providers are: 'local', 'openai', 'huggingface'."
        )

    _provider_cache[cache_key] = instance
    return instance


def embed_texts(texts: List[str], model: Union[BaseEmbeddingProvider, None] = None) -> List[List[float]]:
    """Embed a list of texts using the configured provider or custom model instance."""
    if model is None:
        model = get_embed_model()

    if hasattr(model, "embed_texts"):
        return model.embed_texts(texts)
    elif hasattr(model, "encode"):
        res = model.encode(texts)
        return res.tolist() if isinstance(res, np.ndarray) else res
    else:
        raise AttributeError("Provided model does not have 'embed_texts' or 'encode' method.")


def embed_query(text: str, model: Union[BaseEmbeddingProvider, None] = None) -> List[float]:
    """Embed a single query text using the configured provider or custom model instance."""
    if model is None:
        model = get_embed_model()

    if hasattr(model, "embed_query"):
        return model.embed_query(text)
    return embed_texts([text], model=model)[0]
