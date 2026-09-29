"""Pinecone backend.

Chunks are stored as ``{doc_id}:{index:06d}`` with metadata; each document also gets a
marker record ``__doc__:{doc_id}`` (record_type="doc") so documents can be listed.
Pinecone has no "scan by filter", so chunks are fetched by listing the id prefix.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ..models import Chunk, DocInfo, Hit
from .base import StoreMismatch, VectorStore

_MAX_TEXT_BYTES = 30_000  # Pinecone metadata limit is 40KB per record
_DOC_PREFIX = "__doc__:"


def _clip(text: str) -> str:
    raw = text.encode("utf-8")
    return text if len(raw) <= _MAX_TEXT_BYTES else raw[:_MAX_TEXT_BYTES].decode("utf-8", "ignore")


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


class PineconeStore(VectorStore):
    def __init__(self, index_name: str, namespace: str = "default", cloud: str = "aws",
                 region: str = "us-east-1", api_key: str | None = None, client=None):
        if client is None:
            try:
                from pinecone import Pinecone
            except ImportError as e:
                raise RuntimeError("pip install docchunker[pinecone]") from e
            key = api_key or os.getenv("PINECONE_API_KEY")
            if not key:
                raise RuntimeError("PINECONE_API_KEY is not set")
            client = Pinecone(api_key=key)
        self.pc = client
        self.index_name, self.namespace, self.cloud, self.region = index_name, namespace, cloud, region
        self.index = None

    def bind(self, embedder_name: str, dim: int) -> None:
        names = {i["name"] if isinstance(i, dict) else i.name for i in self.pc.list_indexes()}
        if self.index_name not in names:
            from pinecone import ServerlessSpec

            self.pc.create_index(
                name=self.index_name, dimension=dim, metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        else:
            have = _get(self.pc.describe_index(self.index_name), "dimension")
            if int(have) != dim:
                raise StoreMismatch(
                    f"Pinecone index {self.index_name!r} has dimension {have}, embedder {embedder_name} needs {dim}."
                )
        self.index = self.pc.Index(self.index_name)

    def _ix(self):
        """Index handle; usable without bind() for read/list/delete on an existing index."""
        if self.index is None:
            names = {i["name"] if isinstance(i, dict) else i.name for i in self.pc.list_indexes()}
            if self.index_name not in names:
                raise KeyError(f"Pinecone index {self.index_name!r} does not exist yet; ingest something first")
            self.index = self.pc.Index(self.index_name)
        return self.index

    def _ns(self) -> dict:
        return {"namespace": self.namespace}

    def has_doc(self, doc_id: str) -> bool:
        return bool(_get(self._ix().fetch(ids=[_DOC_PREFIX + doc_id], **self._ns()), "vectors"))

    def add_doc(self, info: DocInfo, chunks, vectors) -> None:
        self.delete_doc(info.doc_id)
        records = [
            {
                "id": c.id,
                "values": [float(x) for x in v],
                "metadata": {
                    "record_type": "chunk", "doc_id": c.doc_id, "index": c.index, "page": c.page,
                    "kind": c.kind, "start": c.start, "end": c.end, "text": _clip(c.text),
                },
            }
            for c, v in zip(chunks, vectors)
        ]
        dim = len(records[0]["values"]) if records else 1
        marker = [0.0] * dim
        marker[0] = 1.0  # Pinecone rejects all-zero dense vectors
        info_json = json.dumps({"name": info.name, "path": info.path, "n_chunks": info.n_chunks,
                                "n_chars": info.n_chars, "created": info.created or time.time()})
        records.append({"id": _DOC_PREFIX + info.doc_id, "values": marker,
                        "metadata": {"record_type": "doc", "doc_id": info.doc_id, "info": info_json}})
        # the marker goes last: a doc only counts as present once every chunk was written
        for i in range(0, len(records), 100):
            self._ix().upsert(vectors=records[i : i + 100], **self._ns())

    @staticmethod
    def _to_chunk(id_: str, md: dict) -> Chunk:
        return Chunk(id=id_, doc_id=md["doc_id"], index=int(md["index"]), text=md.get("text", ""),
                     start=int(md["start"]), end=int(md["end"]), page=int(md["page"]), kind=md.get("kind", "text"))

    def query(self, vector, k, doc_id=None, kind=None, with_vectors=False) -> list[Hit]:
        flt: dict = {"record_type": {"$eq": "chunk"}}
        if doc_id:
            flt["doc_id"] = {"$eq": doc_id}
        if kind:
            flt["kind"] = {"$eq": kind}
        res = self._ix().query(
            vector=[float(x) for x in vector], top_k=k, filter=flt, include_metadata=True,
            include_values=with_vectors, **self._ns(),
        )
        hits = []
        for m in _get(res, "matches"):
            vec = np.array(_get(m, "values"), dtype=np.float32) if with_vectors else None
            hits.append(Hit(self._to_chunk(_get(m, "id"), _get(m, "metadata")), float(_get(m, "score")), vec))
        return hits

    def _list_ids(self, prefix: str) -> list[str]:
        ids: list[str] = []
        for page in self._ix().list(prefix=prefix, **self._ns()):  # generator of id lists
            ids.extend(page)
        return ids

    def _fetch_all(self, ids: list[str]) -> dict:
        out: dict = {}
        for i in range(0, len(ids), 100):
            out.update(_get(self._ix().fetch(ids=ids[i : i + 100], **self._ns()), "vectors"))
        return out

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        fetched = self._fetch_all(self._list_ids(f"{doc_id}:"))
        return sorted((self._to_chunk(k, _get(v, "metadata")) for k, v in fetched.items()), key=lambda c: c.index)

    def list_docs(self) -> list[DocInfo]:
        docs = []
        for v in self._fetch_all(self._list_ids(_DOC_PREFIX)).values():
            md = _get(v, "metadata")
            info = json.loads(md["info"])
            docs.append(DocInfo(md["doc_id"], info["name"], info["path"], info["n_chunks"],
                                info["n_chars"], info["created"]))
        return sorted(docs, key=lambda d: -d.created)

    def delete_doc(self, doc_id: str) -> bool:
        ids = self._list_ids(f"{doc_id}:")
        marker = _DOC_PREFIX + doc_id
        existed = bool(ids) or self.has_doc(doc_id)
        if existed:
            for i in range(0, len(ids), 1000):
                self._ix().delete(ids=ids[i : i + 1000], **self._ns())
            self._ix().delete(ids=[marker], **self._ns())
        return existed
