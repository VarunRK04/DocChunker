from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


@dataclass
class Settings:
    store: str = "local"
    data_dir: Path = Path(".docchunker")
    embedder: str = "sentence-transformers"
    model: str = ""
    chunk_size: int = 800
    chunk_overlap: int = 100
    pinecone_index: str = "docchunker"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        g = os.getenv
        return cls(
            store=g("DOCCHUNKER_STORE", "local"),
            data_dir=Path(g("DOCCHUNKER_DATA_DIR", ".docchunker")),
            embedder=g("DOCCHUNKER_EMBEDDER", "sentence-transformers"),
            model=g("DOCCHUNKER_MODEL", ""),
            chunk_size=_int("DOCCHUNKER_CHUNK_SIZE", 800),
            chunk_overlap=_int("DOCCHUNKER_CHUNK_OVERLAP", 100),
            pinecone_index=g("PINECONE_INDEX", "docchunker"),
            pinecone_namespace=g("PINECONE_NAMESPACE", "default"),
            pinecone_cloud=g("PINECONE_CLOUD", "aws"),
            pinecone_region=g("PINECONE_REGION", "us-east-1"),
        )
