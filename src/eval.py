"""
Evaluate search quality across different chunking configurations.

Defines test queries with expected keywords, runs them against the index,
and reports accuracy and similarity scores.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from src.pipeline import run_pipeline
from src.search import search

# Each entry: (query, set of keywords — a hit if ANY appears in the top result)
TEST_QUERIES: list[tuple[str, set[str]]] = [
    (
        "What is a bid floor?",
        {"bidfloor", "minimum bid", "bid floor", "floor price"},
    ),
    (
        "How are impressions counted?",
        {"impression", "imp", "impressions", "ad impression"},
    ),
    (
        "What is a deal ID?",
        {"dealid", "deal id", "deal", "private marketplace", "pmp"},
    ),
    (
        "How does the auction type work?",
        {"auction", "first price", "second price", "auction type", "auctiontype", "at"},
    ),
    (
        "What is a bid request?",
        {"bid request", "bidrequest", "bidrequestobject", "request object"},
    ),
    (
        "What video formats are supported?",
        {"video", "mimes", "mime", "mp4", "vast", "video object"},
    ),
    (
        "What is a native ad?",
        {"native", "native ad", "native object", "native markup"},
    ),
    (
        "How is user data handled?",
        {"user", "user object", "buyeruid", "gdpr", "consent", "privacy"},
    ),
    (
        "What is a publisher object?",
        {"publisher", "publisher object", "pub", "seller"},
    ),
    (
        "How are banner ads specified?",
        {"banner", "banner object", "width", "height", "banner ad"},
    ),
]


@dataclass
class EvalResult:
    """Results from one evaluation run."""

    config_label: str
    chunk_size: int
    overlap: int
    num_passages: int
    hits: int
    total: int
    avg_score: float
    per_query: list[tuple[str, bool, float, str]]  # (query, hit, score, snippet)

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 0.0


def run_eval(config_label: str = "default", top_k: int = 1) -> EvalResult:
    """Run all test queries and score them against the current index."""
    from pymilvus import MilvusClient

    client = MilvusClient("milvus_demo.db")
    stats = client.get_collection_stats("passages")
    num_passages = stats["row_count"]

    hits = 0
    total_score = 0.0
    per_query: list[tuple[str, bool, float, str]] = []

    for query, expected_keywords in TEST_QUERIES:
        results = search(query, top_k=top_k)
        if not results:
            per_query.append((query, False, 0.0, "(no results)"))
            continue

        top = results[0]
        text_lower = top.text.lower()
        hit = any(kw.lower() in text_lower for kw in expected_keywords)
        if hit:
            hits += 1
        total_score += top.score
        snippet = top.text[:80].replace("\n", " ")
        per_query.append((query, hit, top.score, snippet))

    return EvalResult(
        config_label=config_label,
        chunk_size=0,
        overlap=0,
        num_passages=num_passages,
        hits=hits,
        total=len(TEST_QUERIES),
        avg_score=total_score / len(TEST_QUERIES),
        per_query=per_query,
    )


def print_eval(result: EvalResult) -> None:
    """Print detailed results for a single eval run."""
    print(f"\n{'=' * 70}")
    print(f"Config: {result.config_label}  |  Passages: {result.num_passages}")
    print(f"Accuracy: {result.hits}/{result.total} ({result.accuracy:.0%})  |  Avg score: {result.avg_score:.4f}")
    print(f"{'=' * 70}")
    for query, hit, score, snippet in result.per_query:
        mark = "HIT " if hit else "MISS"
        print(f"  [{mark}] (score={score:.4f}) {query}")
        print(f"         -> {snippet}...")


def run_experiment(pdf_path: str) -> None:
    """Rebuild index with several configs and compare results."""
    configs = [
        ("500c/100o (baseline)", 500, 100),
        ("300c/50o", 300, 50),
        ("750c/150o", 750, 150),
        ("1000c/200o", 1000, 200),
    ]

    all_results: list[EvalResult] = []

    for label, chunk_size, overlap in configs:
        print(f"\n{'#' * 70}")
        print(f"# Building index: {label} (chunk_size={chunk_size}, overlap={overlap})")
        print(f"{'#' * 70}")

        start = time.time()
        run_pipeline(pdf_path, chunk_size=chunk_size, overlap=overlap, force=True)
        build_time = time.time() - start
        print(f"Index built in {build_time:.1f}s")

        result = run_eval(config_label=label)
        result.chunk_size = chunk_size
        result.overlap = overlap
        print_eval(result)
        all_results.append(result)

    # Comparison table
    print(f"\n\n{'=' * 78}")
    print("COMPARISON TABLE")
    print(f"{'=' * 78}")
    print(f"{'Config':<22} {'Chunks':>7} {'Accuracy':>10} {'Avg Score':>11} {'Hits':>6}")
    print(f"{'-' * 22} {'-' * 7} {'-' * 10} {'-' * 11} {'-' * 6}")
    for r in all_results:
        print(
            f"{r.config_label:<22} {r.num_passages:>7} "
            f"{r.accuracy:>9.0%} {r.avg_score:>11.4f} "
            f"{r.hits:>3}/{r.total}"
        )
    print(f"{'=' * 78}")

    best = max(all_results, key=lambda r: (r.accuracy, r.avg_score))
    print(f"\nBest config: {best.config_label} ({best.accuracy:.0%} accuracy, {best.avg_score:.4f} avg score)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.eval <path/to/document.pdf>")
        sys.exit(1)

    run_experiment(sys.argv[1])
