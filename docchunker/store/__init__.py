from __future__ import annotations

from ..config import Settings
from .base import StoreMismatch, VectorStore


def get_store(settings: Settings) -> VectorStore:
    if settings.store == "local":
        from .local import LocalStore

        return LocalStore(settings.data_dir)
    if settings.store == "pinecone":
        from .pinecone import PineconeStore

        return PineconeStore(settings.pinecone_index, settings.pinecone_namespace,
                             settings.pinecone_cloud, settings.pinecone_region)
    raise ValueError(f"Unknown store {settings.store!r} (local | pinecone)")


__all__ = ["get_store", "VectorStore", "StoreMismatch"]
