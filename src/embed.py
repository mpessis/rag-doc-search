"""
Embed text passages using the pymilvus built-in embedding model.

pymilvus ships a DefaultEmbeddingFunction backed by the all-MiniLM-L6-v2
sentence-transformer model.  No external API key is required; the model is
downloaded once and cached locally by the sentence-transformers library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

if TYPE_CHECKING:
    from src.ingest import Passage

# Singleton – loaded once per process.
_ef: SentenceTransformerEmbeddingFunction | None = None


def get_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> SentenceTransformerEmbeddingFunction:
    """Return a (cached) :class:`SentenceTransformerEmbeddingFunction`.

    Args:
        model_name: Any sentence-transformers model name accepted by pymilvus.
        device: ``"cpu"`` or ``"cuda"`` / ``"mps"`` for GPU acceleration.
    """
    global _ef
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=model_name, device=device)
    return _ef


def embed_passages(passages: list["Passage"], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of passages and return a parallel list of dense vectors.

    Args:
        passages: Passages produced by :func:`src.ingest.ingest_pdf`.
        batch_size: Number of texts to encode in one forward pass.

    Returns:
        List of float vectors, one per passage, in the same order.
    """
    ef = get_embedding_function()
    texts = [p.text for p in passages]
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = ef.encode_documents(batch)
        vectors.extend(batch_vectors)

    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single natural-language query string.

    Uses :meth:`encode_queries` which may apply query-specific processing
    (e.g. query prefix in E5 models).
    """
    ef = get_embedding_function()
    return ef.encode_queries([query])[0]
