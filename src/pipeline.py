"""
End-to-end pipeline: PDF → chunk → embed → store in Milvus Lite.

Run once to build the index, then use src/search.py (or src.search.search())
to query it interactively.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pymilvus import DataType, MilvusClient

from src.embed import embed_passages, get_embedding_function
from src.ingest import Passage, ingest_pdf

DB_PATH = "milvus_demo.db"
COLLECTION_NAME = "passages"


def _get_vector_dim() -> int:
    """Return the output dimension of the current embedding model."""
    ef = get_embedding_function()
    return ef.dim


def _create_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """Create a Milvus collection with the required schema."""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("page", DataType.INT32)
    schema.add_field("chunk_index", DataType.INT32)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def _insert_passages(
    client: MilvusClient,
    collection_name: str,
    passages: list[Passage],
    vectors: list[list[float]],
    batch_size: int = 256,
) -> int:
    """Insert passages and their vectors into Milvus in batches.

    Returns:
        Total number of rows inserted.
    """
    total = 0
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        rows = [
            {
                "vector": vec,
                "text": p.text,
                "source": p.source,
                "page": p.page,
                "chunk_index": p.chunk_index,
            }
            for p, vec in zip(batch_passages, batch_vectors)
        ]

        result = client.insert(collection_name=collection_name, data=rows)
        total += result["insert_count"]
        print(f"  Inserted batch {i // batch_size + 1}: {result['insert_count']} rows")

    return total


def run_pipeline(
    pdf_path: str | Path,
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 500,
    overlap: int = 100,
    force: bool = False,
) -> None:
    """Ingest a PDF and populate a Milvus Lite database.

    Args:
        pdf_path: Path to the source PDF.
        db_path: Path where the Milvus Lite ``*.db`` file will be written.
        collection_name: Name of the Milvus collection to create / reuse.
        chunk_size: Target character length per passage.
        overlap: Character overlap between consecutive passages.
        force: If ``True``, drop and recreate the collection even if it exists.
    """
    pdf_path = Path(pdf_path)
    print(f"[1/4] Ingesting PDF: {pdf_path}")
    passages = ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    print(f"      {len(passages)} passages extracted")

    print("[2/4] Embedding passages…")
    vectors = embed_passages(passages)
    print(f"      {len(vectors)} vectors produced (dim={len(vectors[0])})")

    print(f"[3/4] Connecting to Milvus Lite: {db_path}")
    client = MilvusClient("milvus_demo.db")

    if client.has_collection(collection_name):
        if force:
            print(f"      Dropping existing collection '{collection_name}'")
            client.drop_collection(collection_name)
        else:
            print(
                f"      Collection '{collection_name}' already exists. "
                "Pass force=True or delete the .db file to rebuild."
            )
            return

    dim = len(vectors[0])
    print(f"      Creating collection '{collection_name}' (dim={dim})")
    _create_collection(client, collection_name, dim)

    print("[4/4] Inserting passages…")
    total = _insert_passages(client, collection_name, passages, vectors)
    print(f"\nDone. {total} passages stored in '{db_path}'.")
    print("Run a query with:  python -m src.search 'your question here'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <path/to/document.pdf> [--force]")
        sys.exit(1)

    path = sys.argv[1]
    force_flag = "--force" in sys.argv
    run_pipeline(path, force=force_flag)
