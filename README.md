# DocChunker v2

Ingest documents (PDF, DOCX, TXT, Markdown), split them into overlapping chunks, embed them, and search them
semantically. Runs fully offline by default (SQLite + a local embedding model); Pinecone is an optional backend.

v1 (seven loose scripts, fake embeddings) is preserved under the git tag `v1`.

## What changed from v1

| v1 | v2 |
|---|---|
| Character-ordinal "fake embedding" - search was meaningless | Real embeddings (sentence-transformers, OpenAI, or an offline hashing embedder) |
| Zero-vector query + `top_k=1000` filter hack | Real cosine top-k, optional keyword blend and MMR diversification |
| Writer used `test-index`/1024, readers used `text-chunks`/1536 | One config; the index records its embedder and refuses a mismatched one |
| Hardcoded `D:/...pdf`, scripts run on import | CLI + importable package + HTTP API, no hardcoded paths |
| Re-upload duplicated the document | Content-hash doc ids: re-ingesting is a no-op |
| Keyword code/text heuristic (`if`/`for` lines = code) | Line-level scoring; prose that starts with "If..." stays text |
| Chunks lost their page | Chunks never cross pages; each carries page number and exact source offsets |
| No tests | 43 pytest tests |

## Install

```bash
pip install -e ".[st,docx,api,dev]"      # local embeddings, docx, HTTP API, tests
# optional extras: openai, pinecone
```

Copy `.env.example` to `.env` to change defaults (all settings can also be CLI flags).

## Use

```bash
docchunker ingest report.pdf                 # or a folder; add --images-dir imgs/ to extract PDF images
docchunker search "how do I book a vet?" -k 5
docchunker search "vaccination" --doc 3fa9 --kind text --diverse
docchunker docs                              # list indexed documents
docchunker show 3fa9 --limit 3               # inspect chunks (ids can be abbreviated)
docchunker reconstruct 3fa9 -o report.txt    # rebuild the text from its chunks
docchunker delete 3fa9
docchunker serve --port 8000                 # HTTP API
```

Use `--embedder hashing` for an instant, download-free lexical embedder (good for demos and tests).
Use `--store pinecone` with `PINECONE_API_KEY` to use Pinecone (the index is created on first use).

### HTTP API

`POST /ingest` (multipart upload), `GET /search?q=&k=&doc=&kind=&hybrid=&diverse=`, `GET /docs-index`,
`GET /docs-index/{id}/text`, `DELETE /docs-index/{id}`, `GET /health`. Interactive docs at `/docs`.

### Python

```python
from docchunker.pipeline import DocChunker
dc = DocChunker()
dc.ingest_path("reports/")
for hit in dc.search("skin infection", k=3):
    print(hit.score, hit.chunk.page, hit.chunk.text[:80])
```

## How it works

1. **Load** (`loaders/`): PDF via PyMuPDF (per page), DOCX, text/Markdown. Newlines normalised.
2. **Chunk** (`chunking.py`): recursive split on paragraph, line, sentence, clause, word boundaries to `chunk_size`
   characters (default 800) with word-aligned overlap (default 100). Chunks are per page and store `start`/`end`
   offsets, so `reconstruct` is exact rather than a lossy join.
3. **Classify** (`classify.py`): each chunk is tagged `text` or `code`, so you can search only prose or only code.
4. **Embed** (`embeddings/`) and **store** (`store/`): vectors are L2-normalised; the local store is a SQLite file
   with exact cosine search (fine up to a few hundred thousand chunks).
5. **Search** (`search.py`): vector similarity, blended 70/30 with an IDF-weighted keyword score over a candidate pool
   (`--no-hybrid` disables it); `--diverse` applies maximal marginal relevance.

## Limitations

- Scanned PDFs (images only) have no extractable text and are rejected; there is no OCR.
- The local store loads the candidate vectors into memory per query; use Pinecone for very large corpora.
- `chunk_size` counts characters, not tokens.
- The Pinecone and OpenAI paths are tested only against fakes, not live services.

## Tests

```bash
pytest -q
```
