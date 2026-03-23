"""
Ingest a PDF and split it into overlapping text passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Passage:
    text: str
    source: str          # original PDF filename
    page: int            # 0-based page number of the first character
    chunk_index: int     # position among all chunks from this document


def _clean(text: str) -> str:
    """Collapse whitespace while preserving paragraph structure."""
    # Normalise newlines: multiple blank lines → single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse intra-line whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split *text* into overlapping windows of roughly *chunk_size* characters.

    Splitting prefers paragraph boundaries (double newlines) over hard cuts so
    that sentence context is preserved wherever possible.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to split at a paragraph boundary within the last 20 % of the window.
        split_pos = text.rfind("\n\n", start + int(chunk_size * 0.8), end)
        if split_pos == -1:
            # Fall back to the nearest sentence end ". "
            split_pos = text.rfind(". ", start + int(chunk_size * 0.8), end)
            if split_pos != -1:
                split_pos += 1  # include the period
        if split_pos == -1:
            # Hard cut
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        start = split_pos - overlap
        if start < 0:
            start = 0

    return chunks


def ingest_pdf(pdf_path: str | Path, chunk_size: int = 500, overlap: int = 100) -> list[Passage]:
    """Read *pdf_path*, extract text page-by-page, and return a list of :class:`Passage` objects.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target character length of each passage.
        overlap: Number of characters shared between consecutive passages.

    Returns:
        Ordered list of passages ready for embedding.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    source = pdf_path.name

    passages: list[Passage] = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean(raw)
        if not cleaned:
            continue

        for chunk in chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap):
            passages.append(
                Passage(
                    text=chunk,
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return passages
