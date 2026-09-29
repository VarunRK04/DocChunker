"""High-level operations shared by the CLI and the API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import chunk_document, make_doc_id, stitch
from .config import Settings
from .embeddings import get_embedder
from .embeddings.base import Embedder
from .loaders import SUPPORTED, load_document
from .models import DocInfo, Hit
from .search import blend, mmr
from .store import VectorStore, get_store

BATCH = 64


@dataclass
class IngestResult:
    doc_id: str
    name: str
    n_chunks: int
    n_code_chunks: int
    skipped: bool = False  # already indexed


class DocChunker:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None,
                 embedder: Embedder | None = None):
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or get_store(self.settings)
        self.store.bind(self.embedder.name, self.embedder.dim)

    # ---- ingest -------------------------------------------------------------------------
    def ingest_file(self, path: str | Path, force: bool = False) -> IngestResult:
        p = Path(path)
        doc_id = make_doc_id(p.read_bytes())  # content hash => re-ingesting the same file is a no-op
        if not force and self.store.has_doc(doc_id):
            info = next((d for d in self.store.list_docs() if d.doc_id == doc_id), None)
            return IngestResult(doc_id, p.name, info.n_chunks if info else 0, 0, skipped=True)
        doc = load_document(p)
        chunks = chunk_document(doc, doc_id, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(f"{p.name}: no extractable text (scanned PDF? OCR is not supported)")
        import numpy as np

        vecs = np.vstack([self.embedder.embed([c.text.strip() for c in chunks[i : i + BATCH]])
                          for i in range(0, len(chunks), BATCH)])
        info = DocInfo(doc_id, doc.name, str(p.resolve()), len(chunks), len(doc.text), time.time())
        self.store.add_doc(info, chunks, vecs)
        return IngestResult(doc_id, doc.name, len(chunks), sum(c.kind == "code" for c in chunks))

    def ingest_path(self, path: str | Path, force: bool = False,
                    on_error: Callable[[Path, Exception], None] | None = None) -> list[IngestResult]:
        p = Path(path)
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        if not files:
            raise FileNotFoundError(f"No supported files found at {p}")
        results = []
        for f in files:
            try:
                results.append(self.ingest_file(f, force))
            except Exception as e:  # one bad file must not abort a folder ingest
                if on_error is None or p.is_file():
                    raise
                on_error(f, e)
        return results

    # ---- query --------------------------------------------------------------------------
    def search(self, query: str, k: int = 5, doc_id: str | None = None, kind: str | None = None,
               hybrid: bool = True, diverse: bool = False) -> list[Hit]:
        if not query.strip():
            raise ValueError("empty query")
        if doc_id:
            doc_id = self.store.resolve_doc_id(doc_id)
        pool = max(k * 5, 20) if (hybrid or diverse) else k
        hits = self.store.query(self.embedder.embed_query(query), pool, doc_id, kind, with_vectors=diverse)
        if hybrid:
            hits = blend(query, hits)
        if diverse:
            return mmr(hits, k)
        return hits[:k]

    def reconstruct(self, doc_id: str) -> str:
        doc_id = self.store.resolve_doc_id(doc_id)
        return stitch(self.store.get_chunks(doc_id))

    def documents(self) -> list[DocInfo]:
        return self.store.list_docs()

    def delete(self, doc_id: str) -> bool:
        return self.store.delete_doc(self.store.resolve_doc_id(doc_id))
