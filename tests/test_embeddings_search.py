import numpy as np

from docchunker.embeddings.hashing import HashingEmbedder
from docchunker.models import Chunk, Hit
from docchunker.search import blend, keyword_scores, mmr


def _hit(text, score, vec=None, i=0):
    return Hit(Chunk(f"c{i}", "d", i, text, 0, len(text), 1), score, vec)


def test_hashing_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = e.embed(["hello world"]), e.embed(["hello world"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1) < 1e-5
    assert e.embed([""]).shape == (1, 128)  # empty text: zero vector, no NaN
    assert not np.isnan(e.embed([""])).any()


def test_similar_text_scores_higher():
    e = HashingEmbedder()
    q, near, far = e.embed(["dog skin infection", "infection on dog skin", "bake bread"])
    assert q @ near > q @ far


def test_keyword_scores_and_blend_promotes_exact_terms():
    hits = [_hit("nothing relevant here", 0.60, i=0), _hit("the vaccination schedule for puppies", 0.55, i=1)]
    scores = keyword_scores("puppies vaccination", hits)
    assert scores[1] > scores[0]
    assert blend("vaccination schedule", hits, alpha=0.3)[0].chunk.id == "c1"


def test_keyword_scores_empty_query():
    assert keyword_scores("the a of", [_hit("x", 1)]) == [0.0]


def test_mmr_skips_near_duplicates():
    v = np.array
    hits = [_hit("a", 0.9, v([1, 0, 0]), 0), _hit("a2", 0.89, v([0.999, 0.01, 0]), 1), _hit("b", 0.5, v([0, 1, 0]), 2)]
    picked = mmr(hits, 2, lam=0.5)
    assert [h.chunk.id for h in picked] == ["c0", "c2"]
