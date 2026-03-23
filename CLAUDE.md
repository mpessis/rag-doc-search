# CLAUDE.md – Semantic Document Search

## Project overview

Semantic search over technical documentation using a RAG-style retrieval pipeline powered by Milvus Lite.  The demo dataset is the **OpenRTB 2.6 specification** PDF (user-supplied, not stored in the repository).  Given a natural-language question, the system retrieves the most relevant passages from the document.

## Architecture

```
PDF file
  └─► src/ingest.py   – extract text, split into ~500-char overlapping passages
        └─► src/embed.py    – encode with all-MiniLM-L6-v2 (no API key needed)
              └─► MilvusClient (Lite, local .db file)
                    └─► src/search.py  – embed query, cosine search, return top-K
```

`src/pipeline.py` ties ingest → embed → store into a single entry point.

## Python conventions

- **Virtual environment** always in `.venv`; activate before running anything.
- **Type hints** on all function signatures.
- **Docstrings** on every public function and module.
- `from __future__ import annotations` for forward-reference compatibility.
- No `print` inside library functions; only in `__main__` blocks and `pipeline.py` progress messages.

## Key files

| File | Role |
|------|------|
| `src/ingest.py` | `ingest_pdf()` → `list[Passage]` |
| `src/embed.py` | `embed_passages()`, `embed_query()` |
| `src/search.py` | `search()` → `list[SearchResult]`; runnable as `python -m src.search` |
| `src/pipeline.py` | `run_pipeline()`; runnable as `python -m src.pipeline` |
| `milvus_docs.db` | Milvus Lite database (gitignored, created at runtime) |

## Do NOT

- Add a web UI, REST API, or server process.
- Use Docker or any container infrastructure.
- Commit `.env` files, API keys, or the PDF to git.
- Add LLM generation (this is retrieval only — no answer synthesis).
- Over-engineer: no abstract base classes, plugin registries, or config frameworks.
- Switch away from `MilvusClient` with a local file path (no `pymilvus.connections`).

## Running the pipeline

```bash
source .venv/bin/activate
python -m src.pipeline path/to/openrtb2.6.pdf
python -m src.search "What is a bid floor?"
```

## Dependencies

- `pymilvus[model]` – client + `SentenceTransformerEmbeddingFunction`
- `pypdf` – PDF text extraction
- `python-dotenv` – `.env` loading for optional config overrides
