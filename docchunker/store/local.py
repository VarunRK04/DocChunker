"""Single-file SQLite store; vectors are float32 blobs, search is exact cosine via numpy."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from ..models import Chunk, DocInfo, Hit
from .base import StoreMismatch, VectorStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY, name TEXT, path TEXT, n_chunks INTEGER, n_chars INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    idx INTEGER, page INTEGER, kind TEXT, start INTEGER, "end" INTEGER, text TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, idx);
"""

_COLS = 'id, doc_id, idx, page, kind, start, "end", text'


def _chunk(r) -> Chunk:
    return Chunk(id=r[0], doc_id=r[1], index=r[2], page=r[3], kind=r[4], start=r[5], end=r[6], text=r[7])


class LocalStore(VectorStore):
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "index.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def bind(self, embedder_name: str, dim: int) -> None:
        have = self._meta("embedder")
        if have is None:
            with self.db:
                self.db.executemany(
                    "INSERT INTO meta VALUES (?,?)", [("embedder", embedder_name), ("dim", str(dim))]
                )
        elif have != embedder_name or int(self._meta("dim")) != dim:
            raise StoreMismatch(
                f"Index was built with {have} (dim {self._meta('dim')}), but {embedder_name} (dim {dim}) "
                "was requested. Use the same embedder or a fresh --data-dir."
            )

    def has_doc(self, doc_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def add_doc(self, info: DocInfo, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        with self.db:  # one transaction: all or nothing
            self.db.execute("DELETE FROM docs WHERE doc_id=?", (info.doc_id,))
            self.db.execute(
                "INSERT INTO docs VALUES (?,?,?,?,?,?)",
                (info.doc_id, info.name, info.path, info.n_chunks, info.n_chars, info.created or time.time()),
            )
            self.db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (c.id, c.doc_id, c.index, c.page, c.kind, c.start, c.end, c.text, v.tobytes())
                    for c, v in zip(chunks, vectors)
                ],
            )

    def query(self, vector, k, doc_id=None, kind=None, with_vectors=False) -> list[Hit]:
        where, args = [], []
        if doc_id:
            where.append("doc_id=?")
            args.append(doc_id)
        if kind:
            where.append("kind=?")
            args.append(kind)
        sql = f"SELECT {_COLS}, vec FROM chunks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self.db.execute(sql, args).fetchall()
        if not rows or k < 1:
            return []
        q = np.asarray(vector, dtype=np.float32)
        mat = np.frombuffer(b"".join(r[8] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        if mat.shape[1] != q.shape[0]:
            raise StoreMismatch(f"Query dimension {q.shape[0]} != index dimension {mat.shape[1]}")
        scores = mat @ q
        k = min(k, len(rows))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(_chunk(rows[i]), float(scores[i]), mat[i] if with_vectors else None) for i in top]

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self.db.execute(f"SELECT {_COLS} FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)).fetchall()
        return [_chunk(r) for r in rows]

    def list_docs(self) -> list[DocInfo]:
        rows = self.db.execute(
            "SELECT doc_id, name, path, n_chunks, n_chars, created FROM docs ORDER BY created DESC"
        ).fetchall()
        return [DocInfo(*r) for r in rows]

    def delete_doc(self, doc_id: str) -> bool:
        with self.db:
            return self.db.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,)).rowcount > 0
