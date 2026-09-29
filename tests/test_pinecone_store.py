"""PineconeStore against an in-memory fake that mimics the client surface we use."""
import numpy as np
import pytest

from conftest import CORPUS
from docchunker.config import Settings
from docchunker.embeddings.hashing import HashingEmbedder
from docchunker.pipeline import DocChunker
from docchunker.store.base import StoreMismatch
from docchunker.store.pinecone import PineconeStore


class FakeIndex:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def upsert(self, vectors, namespace):
        assert len(vectors) <= 100
        for v in vectors:
            assert any(x != 0 for x in v["values"]), "Pinecone rejects all-zero vectors"
            self.rows[(namespace, v["id"])] = v

    def fetch(self, ids, namespace):
        return {"vectors": {i: self.rows[(namespace, i)] for i in ids if (namespace, i) in self.rows}}

    def list(self, prefix, namespace):
        ids = sorted(i for ns, i in self.rows if ns == namespace and i.startswith(prefix))
        for k in range(0, len(ids), 3):  # small pages to exercise pagination
            yield ids[k : k + 3]

    def delete(self, ids, namespace):
        for i in ids:
            self.rows.pop((namespace, i), None)

    def query(self, vector, top_k, filter, include_metadata, include_values, namespace):
        q = np.array(vector)
        out = []
        for (ns, id_), r in self.rows.items():
            md = r["metadata"]
            if ns != namespace or any(md.get(k) != c["$eq"] for k, c in filter.items()):
                continue
            m = {"id": id_, "score": float(np.array(r["values"]) @ q), "metadata": md}
            if include_values:
                m["values"] = r["values"]
            out.append(m)
        return {"matches": sorted(out, key=lambda m: -m["score"])[:top_k]}


class FakeClient:
    def __init__(self, dim=None):
        self.index = FakeIndex()
        self.dim = dim

    def list_indexes(self):
        return [{"name": "idx"}] if self.dim else []

    def describe_index(self, name):
        return {"dimension": self.dim}

    def Index(self, name):
        return self.index


@pytest.fixture
def pdc(tmp_path):
    emb = HashingEmbedder()
    store = PineconeStore("idx", "ns", client=FakeClient(dim=emb.dim))
    settings = Settings(embedder="hashing", chunk_size=200, chunk_overlap=30)
    return DocChunker(settings, store=store, embedder=emb)


def test_full_cycle(pdc, corpus):
    results = pdc.ingest_path(corpus)
    assert {d.name for d in pdc.documents()} == {"pets.txt", "cooking.txt"}
    assert pdc.ingest_file(corpus / "pets.txt").skipped

    hits = pdc.search("sourdough starter", k=3, diverse=True)
    assert hits[0].chunk.text.startswith("To bake")
    assert all(h.chunk.doc_id != "__doc__" for h in hits)  # marker records never leak into results

    pets = next(r for r in results if r.name == "pets.txt")
    assert pdc.reconstruct(pets.doc_id) == CORPUS["pets.txt"].strip()
    assert pdc.delete(pets.doc_id)
    with pytest.raises(KeyError):
        pdc.delete(pets.doc_id)
    assert pdc.store.delete_doc(pets.doc_id) is False
    assert [d.name for d in pdc.documents()] == ["cooking.txt"]
    assert pdc.store.index.rows and all(pets.doc_id not in i for _, i in pdc.store.index.rows)


def test_dimension_mismatch():
    store = PineconeStore("idx", client=FakeClient(dim=1536))
    with pytest.raises(StoreMismatch, match="1536"):
        store.bind("hashing-512", 512)


def test_long_text_is_clipped_under_metadata_limit(pdc):
    from docchunker.models import Chunk, DocInfo

    big = "é" * 40_000
    c = Chunk("d:000000", "d", 0, big, 0, len(big), 1)
    pdc.store.add_doc(DocInfo("d", "n", "p", 1, len(big), 1.0), [c], np.ones((1, 512), dtype=np.float32))
    stored = pdc.store.index.rows[("ns", "d:000000")]["metadata"]["text"]
    assert len(stored.encode()) <= 30_000


def test_read_operations_work_without_bind():
    client = FakeClient(dim=512)
    store = PineconeStore("idx", "ns", client=client)
    assert store.list_docs() == []  # no bind() (and so no embedder) needed to list
    missing = PineconeStore("nope", "ns", client=FakeClient(dim=None))
    with pytest.raises(KeyError, match="does not exist"):
        missing.list_docs()
