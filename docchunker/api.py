"""Thin HTTP layer over DocChunker. Uploads only; the server never reads arbitrary local paths."""
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .loaders import SUPPORTED
from .pipeline import DocChunker

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(settings: Settings | None = None, dc: DocChunker | None = None):
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile

    app = FastAPI(title="DocChunker", version="2.0.0")
    state: dict = {"dc": dc}
    lock = threading.Lock()  # one SQLite connection: serialise access across worker threads

    def svc() -> DocChunker:
        if state["dc"] is None:  # lazy: loading the embedding model is slow
            state["dc"] = DocChunker(settings)
        return state["dc"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # plain `def` endpoints run in FastAPI's threadpool, so slow embedding never blocks the event loop
    @app.post("/ingest")
    def ingest(file: UploadFile = File(...), force: bool = False):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(415, f"Unsupported type {suffix!r}; use one of {', '.join(SUPPORTED)}")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / Path(file.filename).name  # basename only: no path traversal
            target.write_bytes(data)
            try:
                with lock:
                    r = svc().ingest_file(target, force)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
        return asdict(r)

    @app.get("/search")
    def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50), doc: str | None = None,
               kind: str | None = Query(None, pattern="^(text|code)$"), hybrid: bool = True, diverse: bool = False):
        try:
            with lock:
                hits = svc().search(q, k, doc, kind, hybrid, diverse)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return [{"score": h.score, **asdict(h.chunk)} for h in hits]

    @app.get("/docs-index")
    def docs():
        with lock:
            return [asdict(d) for d in svc().documents()]

    @app.get("/docs-index/{doc_id}/text")
    def text(doc_id: str):
        try:
            with lock:
                return {"doc_id": doc_id, "text": svc().reconstruct(doc_id)}
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e

    @app.delete("/docs-index/{doc_id}")
    def delete(doc_id: str):
        try:
            with lock:
                svc().delete(doc_id)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        return {"deleted": doc_id}

    return app
