# Semantic Document Search

A lightweight semantic search pipeline for querying technical documentation using natural language. Demonstrated with the OpenRTB 2.6 programmatic advertising specification.

## Why This Matters

Dense technical documentation is everywhere — protocol specs, compliance policies, API references, onboarding guides — and keyword search consistently fails it. The right answer is buried in a paragraph that doesn't share a single word with your query.

This pipeline lets anyone ask questions in plain English and get back the exact passages that answer them:

- **Solutions Engineers** answering client integration questions without reading 100-page specs
- **Trust & Safety teams** searching policy documentation for specific scenarios or edge cases
- **AdTech teams** navigating protocol specs like OpenRTB, VAST, or SKAN
- **Any team** with internal documentation too dense to Ctrl+F effectively

No external API, no server, no infrastructure. Runs entirely on a laptop.

## How it works

```
PDF → chunk (500 chars, 100 overlap) → embed (all-MiniLM-L6-v2) → Milvus Lite → cosine search
```

1. **Ingest** – `pypdf` extracts text page by page; text is split into overlapping passages.
2. **Embed** – `pymilvus`'s built-in `SentenceTransformerEmbeddingFunction` encodes every passage as a dense vector (no API key needed; model downloads once).
3. **Store** – Vectors and metadata are written to a local Milvus Lite `.db` file.
4. **Query** – Your question is embedded with the same model and the top-K nearest passages are returned by cosine similarity.

## Example Queries

Querying the OpenRTB 2.6 spec:

```
$ python -m src.search "What is a bid floor?"

--- Result 1 | score=0.6100 | OpenRTB-2-6_FINAL.pdf p.29 ---
bidfloor: Minimum bid for this impression expressed in CPM. Default 0.
...

$ python -m src.search "What fields are required in a bid request?"

--- Result 1 | score=0.6900 | OpenRTB-2-6_FINAL.pdf p.44 ---
The following attributes are required and must be present in every bid request...

$ python -m src.search "What is a private marketplace deal?"

--- Result 1 | score=0.6400 | OpenRTB-2-6_FINAL.pdf p.29 ---
Pmp object: A container for any private marketplace (PMP) deals applicable to
this impression for a programmatic guaranteed or private auction...
```

Relevant results at scores above 0.60 — without knowing the exact field name or page number upfront.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Build the index

```bash
python -m src.pipeline ~/Downloads/openrtb2.6.pdf
```

This creates `milvus_demo.db` in the current directory. Run with `--force` to rebuild from scratch.

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

## Adapt to Your Documentation

The demo uses OpenRTB 2.6, but the pipeline is document-agnostic. Point it at any PDF:

- Compliance and policy documents
- Internal API or SDK references
- Product specs or onboarding guides
- Legal contracts or regulatory filings

One command to index, one command to search.

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
