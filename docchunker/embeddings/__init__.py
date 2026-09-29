from __future__ import annotations

from ..config import Settings
from .base import Embedder


def get_embedder(settings: Settings) -> Embedder:
    name = settings.embedder.lower()
    if name in ("hashing", "hash"):
        from .hashing import HashingEmbedder

        return HashingEmbedder()
    if name in ("sentence-transformers", "st", "local"):
        from .st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.model or "all-MiniLM-L6-v2")
    if name == "openai":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder(settings.model or "text-embedding-3-small")
    raise ValueError(f"Unknown embedder {settings.embedder!r} (hashing | sentence-transformers | openai)")
