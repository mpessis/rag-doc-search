# Semantic Document Search

A lightweight semantic search pipeline for querying technical documentation using natural language. Demonstrated with the OpenRTB 2.6 programmatic advertising specification.

## How it works

```
PDF → chunk (500 chars, 100 overlap) → embed (all-MiniLM-L6-v2) → Milvus Lite → cosine search
```

1. **Ingest** – `pypdf` extracts text page by page; text is split into overlapping passages.
2. **Embed** – `pymilvus`'s built-in `SentenceTransformerEmbeddingFunction` encodes every passage as a dense vector (no API key needed; model downloads once).
3. **Store** – Vectors and metadata are written to a local Milvus Lite `.db` file.
4. **Query** – Your question is embedded with the same model and the top-K nearest passages are returned.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Build the index

Place your PDF in a convenient location (e.g. `~/Downloads/openrtb2.6.pdf`) and run:

```bash
python -m src.pipeline ~/Downloads/openrtb2.6.pdf
```

This creates `milvus_docs.db` in the current directory. Run with `--force` to rebuild from scratch.

### 2. Search

```bash
python -m src.search "What is the difference between a bid request and a bid response?"
python -m src.search "How are native ads represented in OpenRTB?"
python -m src.search "What fields are required in an impression object?"
```

### 3. Use from Python

```python
from src.pipeline import run_pipeline
from src.search import search, print_results

# Build index (only once)
run_pipeline("~/Downloads/openrtb2.6.pdf")

# Query
results = search("What is a seat ID?", top_k=5)
print_results(results)
```

## Project layout

```
src/
  ingest.py    # PDF → list[Passage]
  embed.py     # list[Passage] → list[vector]
  search.py    # query string → list[SearchResult]
  pipeline.py  # orchestrates ingest → embed → store
requirements.txt
```

## Notes

- The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from HuggingFace via `sentence-transformers`.
- Milvus Lite stores everything in a single `.db` file; no server process is needed.
- The PDF is **not** stored in this repository. Point the pipeline at your own copy.
