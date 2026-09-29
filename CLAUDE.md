# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[st,docx,api,dev]"
python -m pytest -q                         # all tests (uses --basetemp=.pytest_tmp; the default Windows temp dir was not writable)
python -m pytest tests/test_chunking.py::test_roundtrip_exact
docchunker <ingest|search|docs|show|reconstruct|delete|serve> ...   # also: python -m docchunker.cli
```

Tests use the offline `hashing` embedder and temp SQLite stores; nothing needs network or API keys.

## Architecture

`pipeline.DocChunker` is the only orchestration layer; `cli.py` and `api.py` are thin wrappers over it.
Flow: `loaders` -> `chunking` (+`classify`) -> `embeddings` -> `store`; queries go `embeddings` -> `store.query` -> `search` (keyword blend / MMR).

Invariants that span files:
- **Doc id = sha256 of file bytes [:16]**; chunk id = `{doc_id}:{index:06d}`. Re-ingesting identical content is a no-op (`ingest_file` checks `store.has_doc`).
- **Chunks carry `start`/`end` offsets into `Document.text`** (pages joined by `PAGE_SEP`). `chunking.stitch` uses them to rebuild text exactly; interior chunks deliberately keep raw whitespace, only the document's first/last chunk are trimmed. Embedding uses `text.strip()`.
- **Chunking is per page**, so a chunk never straddles pages (accurate `page` citations). Overlap never crosses a page.
- **Stores are pluggable** (`store/base.VectorStore`): `local` (SQLite + numpy exact cosine) and `pinecone`. A store records the embedder name/dim on first `bind()` and raises `StoreMismatch` afterwards for a different one. The Pinecone store keeps a `__doc__:{id}` marker record per document (written last) for listing, and filters queries to `record_type=chunk`.
- Embedders return L2-normalised float32 so dot product = cosine.
- `api.py` must not use `from __future__ import annotations` (FastAPI resolves annotations of routes defined inside `create_app`).

The Pinecone and OpenAI paths have only been tested against fakes (`tests/test_pinecone_store.py`).
v1 (loose scripts with a fake embedding) is at git tag `v1`.
