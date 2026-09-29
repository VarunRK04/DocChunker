"""Deterministic, dependency-free lexical embedder (signed feature hashing).

Not semantic, but similar wording gives similar vectors, which is enough for
offline use and fast tests.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .base import Embedder, normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str):
        words = _TOKEN.findall(text.lower())
        yield from words
        yield from (f"{a} {b}" for a, b in zip(words, words[1:]))

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return normalize(out)
