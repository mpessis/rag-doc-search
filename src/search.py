"""
Query the Milvus Lite collection and return ranked passages.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymilvus import MilvusClient

from src.embed import embed_query


@dataclass
class SearchResult:
    text: str
    source: str
    page: int
    chunk_index: int
    score: float


def search(
    query: str,
    db_path: str = "milvus_demo.db",
    collection_name: str = "passages",
    top_k: int = 5,
) -> list[SearchResult]:
    """Run a semantic search against a Milvus Lite collection.

    Args:
        query: Natural-language question or phrase.
        db_path: Path to the Milvus Lite ``*.db`` file written by the pipeline.
        collection_name: Name of the collection to query.
        top_k: Number of top results to return.

    Returns:
        List of :class:`SearchResult` objects ordered by descending relevance.
    """
    client = MilvusClient("milvus_demo.db")

    query_vector = embed_query(query)

    raw = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "page", "chunk_index"],
    )

    results: list[SearchResult] = []
    for hit in raw[0]:
        entity = hit["entity"]
        results.append(
            SearchResult(
                text=entity["text"],
                source=entity["source"],
                page=entity["page"],
                chunk_index=entity["chunk_index"],
                score=hit["distance"],
            )
        )

    return results


def print_results(results: list[SearchResult]) -> None:
    """Pretty-print search results to stdout."""
    if not results:
        print("No results found.")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\n--- Result {rank} | score={r.score:.4f} | {r.source} p.{r.page + 1} ---")
        print(r.text)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is an impression?"
    print(f"Query: {query}\n")
    print_results(search(query))
