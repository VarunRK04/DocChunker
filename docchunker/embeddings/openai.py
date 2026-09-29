from __future__ import annotations

import numpy as np

from .base import Embedder, normalize

_KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install docchunker[openai]") from e
        self._client = OpenAI()  # reads OPENAI_API_KEY
        self._model = model
        self.name = f"openai:{model}"
        self.dim = _KNOWN_DIMS.get(model) or len(self.embed(["dim probe"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), 96):
            resp = self._client.embeddings.create(model=self._model, input=texts[i : i + 96])
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return normalize(np.array(rows))
