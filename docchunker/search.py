"""Re-ranking on top of vector hits: keyword-coverage blending and MMR diversification."""
from __future__ import annotations

import math
import re

import numpy as np

from .models import Hit

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)


def _terms(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def keyword_scores(query: str, hits: list[Hit]) -> list[float]:
    """IDF-weighted share of query terms present in each hit (idf over the candidate pool)."""
    q = set(_terms(query))
    if not q or not hits:
        return [0.0] * len(hits)
    bags = [set(_terms(h.chunk.text)) for h in hits]
    n = len(hits)
    idf = {t: math.log(1 + n / (1 + sum(t in b for b in bags))) for t in q}
    total = sum(idf.values()) or 1.0
    return [sum(idf[t] for t in q if t in b) / total for b in bags]


def blend(query: str, hits: list[Hit], alpha: float = 0.7) -> list[Hit]:
    """score = alpha * vector + (1 - alpha) * keyword; returns hits re-sorted."""
    kw = keyword_scores(query, hits)
    blended = [Hit(h.chunk, alpha * h.score + (1 - alpha) * s, h.vector) for h, s in zip(hits, kw)]
    return sorted(blended, key=lambda h: -h.score)


def mmr(hits: list[Hit], k: int, lam: float = 0.7) -> list[Hit]:
    """Maximal marginal relevance: trade relevance against similarity to already-picked hits."""
    if any(h.vector is None for h in hits):
        raise ValueError("MMR needs hit vectors")
    remaining = list(range(len(hits)))
    vecs = np.array([h.vector for h in hits], dtype=np.float32)
    picked: list[int] = []
    while remaining and len(picked) < k:
        if not picked:
            best = max(remaining, key=lambda i: hits[i].score)
        else:
            sim = vecs[remaining] @ vecs[picked].T  # (remaining, picked)
            gain = [lam * hits[i].score - (1 - lam) * float(sim[r].max()) for r, i in enumerate(remaining)]
            best = remaining[int(np.argmax(gain))]
        picked.append(best)
        remaining.remove(best)
    return [hits[i] for i in picked]
