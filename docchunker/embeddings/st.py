from __future__ import annotations

import numpy as np

from .base import Embedder, normalize


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("pip install docchunker[st]  (or use --embedder hashing)") from e
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return normalize(vecs)
