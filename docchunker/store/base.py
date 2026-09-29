from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import Chunk, DocInfo, Hit


class StoreMismatch(RuntimeError):
    """The index was built with a different embedder / dimension."""


class VectorStore(ABC):
    @abstractmethod
    def bind(self, embedder_name: str, dim: int) -> None:
        """Record (or verify) which embedder this index uses. Raises StoreMismatch."""

    @abstractmethod
    def has_doc(self, doc_id: str) -> bool: ...

    @abstractmethod
    def add_doc(self, info: DocInfo, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Insert a document and all its chunks (replacing any previous version)."""

    @abstractmethod
    def query(
        self, vector: np.ndarray, k: int, doc_id: str | None = None, kind: str | None = None,
        with_vectors: bool = False,
    ) -> list[Hit]: ...

    @abstractmethod
    def get_chunks(self, doc_id: str) -> list[Chunk]: ...

    @abstractmethod
    def list_docs(self) -> list[DocInfo]: ...

    @abstractmethod
    def delete_doc(self, doc_id: str) -> bool: ...

    def resolve_doc_id(self, prefix: str) -> str:
        """Allow unambiguous prefixes of a doc id."""
        ids = [d.doc_id for d in self.list_docs() if d.doc_id.startswith(prefix)]
        if len(ids) == 1:
            return ids[0]
        raise KeyError(f"No document with id {prefix!r}" if not ids else f"Ambiguous id {prefix!r}: {ids}")
