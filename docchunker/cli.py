from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .store.base import StoreMismatch


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docchunker", description="Ingest, chunk, embed and search documents.")
    p.add_argument("--store", choices=["local", "pinecone"], help="vector store backend")
    p.add_argument("--data-dir", help="local store directory")
    p.add_argument("--embedder", help="hashing | sentence-transformers | openai")
    p.add_argument("--model", help="embedding model name")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index a file or a folder")
    s.add_argument("path")
    s.add_argument("--force", action="store_true", help="re-index even if already present")
    s.add_argument("--images-dir", help="also extract PDF images to this folder")

    s = sub.add_parser("search", help="semantic search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--doc", help="restrict to a document id (prefix ok)")
    s.add_argument("--kind", choices=["text", "code"])
    s.add_argument("--no-hybrid", action="store_true", help="vector score only")
    s.add_argument("--diverse", action="store_true", help="MMR re-ranking to reduce near-duplicates")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("docs", help="list indexed documents")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("show", help="print the chunks of a document")
    s.add_argument("doc_id")
    s.add_argument("--limit", type=int, default=0, help="max chunks (0 = all)")

    s = sub.add_parser("reconstruct", help="rebuild the original text from its chunks")
    s.add_argument("doc_id")
    s.add_argument("-o", "--output", help="write to a file instead of stdout")

    s = sub.add_parser("delete", help="remove a document")
    s.add_argument("doc_id")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def _settings(a: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if a.store:
        s.store = a.store
    if a.data_dir:
        s.data_dir = Path(a.data_dir)
    if a.embedder:
        s.embedder = a.embedder
    if a.model:
        s.model = a.model
    if a.chunk_size:
        s.chunk_size = a.chunk_size
    if a.chunk_overlap is not None:
        s.chunk_overlap = a.chunk_overlap
    return s


def _emit(text: str) -> None:
    print(text)


def _run(a: argparse.Namespace) -> int:
    settings = _settings(a)
    if a.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=a.host, port=a.port)
        return 0

    from .pipeline import DocChunker

    dc = DocChunker(settings)
    if a.cmd == "ingest":
        def warn(f, e):
            print(f"! skipped {f}: {e}", file=sys.stderr)

        results = dc.ingest_path(a.path, a.force, on_error=warn)
        for r in results:
            state = "already indexed" if r.skipped else f"{r.n_chunks} chunks ({r.n_code_chunks} code)"
            print(f"{r.doc_id}  {r.name}: {state}")
        if a.images_dir:
            from .loaders.pdf import extract_images

            for f in ([Path(a.path)] if Path(a.path).is_file() else Path(a.path).rglob("*.pdf")):
                if f.suffix.lower() == ".pdf":
                    print(f"{len(extract_images(f, Path(a.images_dir) / f.stem))} images from {f.name}")
    elif a.cmd == "search":
        hits = dc.search(a.query, a.k, a.doc, a.kind, hybrid=not a.no_hybrid, diverse=a.diverse)
        if a.json:
            _emit(json.dumps([{"score": h.score, **asdict(h.chunk)} for h in hits], ensure_ascii=False, indent=2))
        elif not hits:
            print("no results")
        else:
            for i, h in enumerate(hits, 1):
                c = h.chunk
                snippet = " ".join(c.text.split())
                _emit(f"[{i}] {h.score:.3f}  doc {c.doc_id[:8]}  p.{c.page}  #{c.index} ({c.kind})\n    "
                      f"{snippet[:300]}{'...' if len(snippet) > 300 else ''}")
    elif a.cmd == "docs":
        docs = dc.documents()
        if a.json:
            _emit(json.dumps([asdict(d) for d in docs], indent=2))
        else:
            for d in docs:
                print(f"{d.doc_id}  {d.n_chunks:>5} chunks  {d.n_chars:>8} chars  {d.name}")
            if not docs:
                print("no documents")
    elif a.cmd == "show":
        chunks = dc.store.get_chunks(dc.store.resolve_doc_id(a.doc_id))
        for c in chunks[: a.limit or None]:
            _emit(f"--- #{c.index} p.{c.page} {c.kind} [{c.start}:{c.end}]\n{c.text}")
    elif a.cmd == "reconstruct":
        text = dc.reconstruct(a.doc_id)
        if a.output:
            Path(a.output).write_text(text, encoding="utf-8")
            print(f"wrote {len(text)} chars to {a.output}")
        else:
            _emit(text)
    elif a.cmd == "delete":
        print("deleted" if dc.delete(a.doc_id) else "not found")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        return _run(args)
    except (StoreMismatch, KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
