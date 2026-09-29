<div align="center">

# 📄 DocChunker

**Turn documents into searchable knowledge, on your own machine.**

Ingest PDFs, Word files and Markdown → split them into page-aware chunks → embed them → ask questions in plain English.
Offline by default. Pinecone when you want to scale.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)
![store](https://img.shields.io/badge/store-SQLite%20%7C%20Pinecone-informational)

</div>

---

## Why this exists

DocChunker started as an internship project: a handful of scripts that chunked a PDF and pushed it to Pinecone.
It worked as a demo, but the "embeddings" were fake, so search never returned anything meaningful.
**v2 is a ground-up rebuild** that makes retrieval actually work, and turns the scripts into a proper tool.
(The original is preserved under the git tag `v1`.)

## Quick start

```bash
git clone https://github.com/VarunRK04/DocChunker && cd DocChunker
pip install -e ".[st,docx,api,dev]"

docchunker ingest my_report.pdf
docchunker search "how do I book a vet consultation?"
```

```text
[1] 0.319  doc f09e060c  p.2  #1 (text)
    Owners can book a consultation with a nearby veterinarian directly from the app...
```

No API keys, no cloud account. The first run downloads a small local embedding model (`all-MiniLM-L6-v2`).
Want to try it without any download? Add `--embedder hashing`.

## What you get

| | |
|---|---|
| 🔎 **Real semantic search** | Cosine similarity over real embeddings, blended with an IDF-weighted keyword score. `--diverse` removes near-duplicate hits (MMR). |
| 📑 **Page-accurate chunks** | Chunks never straddle pages, so every result cites the right page. Overlap is word-aligned. |
| ♻️ **Lossless reconstruction** | Every chunk stores its exact offsets, so `reconstruct` rebuilds the source text instead of approximating it. |
| 🧩 **Code vs. prose** | Chunks are tagged `text` or `code`; search just one with `--kind code`. |
| 🔁 **Idempotent ingest** | Documents are identified by content hash. Re-ingesting the same file is a no-op. |
| 🛡️ **Mismatch protection** | The index remembers which embedder built it and refuses a different one, instead of silently returning garbage. |
| 🔌 **Pluggable backends** | Stores: local SQLite or Pinecone. Embedders: sentence-transformers, OpenAI, or an offline hashing embedder. |
| 🌐 **Three ways in** | CLI, HTTP API (FastAPI), or plain Python. |

## Usage

### Command line

```bash
docchunker ingest reports/                   # a file or a whole folder (hidden/vendor dirs are skipped)
docchunker ingest paper.pdf --images-dir img # also extract embedded PDF images
docchunker search "skin infection" -k 5
docchunker search "vaccination" --doc 3fa9 --kind text --diverse
docchunker docs                              # what is indexed
docchunker show 3fa9 --limit 3               # inspect chunks (ids can be abbreviated)
docchunker reconstruct 3fa9 -o paper.txt
docchunker delete 3fa9
```

Global flags: `--store`, `--data-dir`, `--embedder`, `--model`, `--chunk-size`, `--chunk-overlap`.
Add `--json` to `search` and `docs` for scripting.

### Python

```python
from docchunker.pipeline import DocChunker

dc = DocChunker()
dc.ingest_path("reports/")
for hit in dc.search("skin infection", k=3):
    print(f"{hit.score:.2f}  p.{hit.chunk.page}  {hit.chunk.text[:80]}")
```

### HTTP API

```bash
docchunker serve --port 8000     # interactive docs at http://127.0.0.1:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | multipart file upload (50 MB cap) |
| `GET` | `/search?q=&k=&doc=&kind=&hybrid=&diverse=` | search |
| `GET` | `/docs-index` | list documents |
| `GET` | `/docs-index/{id}/text` | reconstructed text |
| `DELETE` | `/docs-index/{id}` | remove a document |

The API only accepts uploads; it never reads arbitrary paths from the server's disk.

### Using Pinecone

```bash
pip install -e ".[pinecone]"
export PINECONE_API_KEY=...        # see .env.example for the other settings
docchunker --store pinecone ingest report.pdf
```

The serverless index is created on first use with the right dimension for your embedder.

## How it works

```
 file ──► loaders ──► chunking ──► classify ──► embeddings ──► store
 (pdf/docx/txt/md)   per page,     text | code   L2-normalised   SQLite / Pinecone
                     word-aligned                vectors
                     overlap

 query ─► embeddings ─► store.query (top-k pool) ─► keyword blend ─► MMR (optional) ─► results
```

- **Chunking** splits recursively on paragraph → line → sentence → clause → word boundaries (default 800 chars,
  100 overlap) and records `start`/`end` offsets into the document text.
- **Storage** keeps text, metadata and vectors together. The local store does exact cosine search with numpy.
- **Hybrid ranking** takes a candidate pool (5× `k`), then scores `0.7 × vector + 0.3 × keyword coverage`.

<details>
<summary>Project layout</summary>

```
docchunker/
  loaders/      pdf.py  docx.py  text.py         file → pages
  chunking.py   splitter + exact stitch()
  classify.py   code vs. prose
  embeddings/   st.py  openai.py  hashing.py
  store/        local.py  pinecone.py  base.py
  search.py     keyword blend, MMR
  pipeline.py   DocChunker: ingest / search / reconstruct
  cli.py        api.py
tests/          43 tests, all offline
```
</details>

## Configuration

Everything can be set by flag or environment variable (a `.env` file works). See [`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `DOCCHUNKER_STORE` | `local` | `local` or `pinecone` |
| `DOCCHUNKER_DATA_DIR` | `.docchunker` | where the local index lives |
| `DOCCHUNKER_EMBEDDER` | `sentence-transformers` | or `openai`, `hashing` |
| `DOCCHUNKER_CHUNK_SIZE` / `_OVERLAP` | `800` / `100` | in characters |

## v1 → v2

| v1 | v2 |
|---|---|
| Fake character-code embeddings | Real embeddings; search returns relevant results |
| Zero-vector query + `top_k=1000` filter hack | Proper top-k similarity |
| Writer and readers used different indexes and dimensions | One config, embedder checked on every open |
| Hardcoded `D:/...` path, scripts ran on import | CLI, importable package, HTTP API |
| Re-upload duplicated the document | Content-hash ids |
| Keyword heuristic flagged `if`/`for` prose as code | Line-level scoring |
| No tests | 43 tests |

## Limitations

- Scanned (image-only) PDFs have no extractable text and are rejected; there is no OCR.
- The local store reads candidate vectors into memory per query: comfortable for hundreds of thousands of chunks,
  not millions. Use Pinecone beyond that.
- `chunk_size` counts characters, not tokens.
- The Pinecone and OpenAI paths are tested against fakes only, not live services.

## Development

```bash
pytest -q          # offline: uses the hashing embedder and temporary SQLite stores
```

---

<div align="center">Built by <a href="https://github.com/VarunRK04">Varun</a></div>
