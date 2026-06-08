"""
pdf_loader.py
=============
Pipeline stage 1: turn a PDF file on disk into a list of text `Chunk`s.

Two responsibilities:
  1. Read each page's text out of the PDF (using pypdf).
  2. Split that text into overlapping word-windows so each chunk is small
     enough to embed and retrieve precisely.

Why chunk?
----------
  * Embedding models have an input length limit (a few hundred tokens). One
    huge vector for a whole document loses fine-grained meaning — everything
    "averages out".
  * LLM context windows cost money/time. We want to send only the most
    relevant slice of the document, not the whole thing.

Why per-page chunks (not document-wide)?
----------------------------------------
Splitting page-by-page means every chunk knows exactly which page it came
from. That lets the LLM cite page numbers in its answers — a simple but
powerful UX trick that makes hallucinations easy to spot.

Caveat: scanned PDFs
--------------------
pypdf only extracts TEXT. If the PDF is just images of text (a scan), every
page returns ""; you'd need OCR (e.g. pytesseract) to handle that. We don't
crash — we just return an empty list and let main.py warn the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    """One unit of retrievable text plus the page it came from."""

    text: str
    page: int  # 1-based page number in the source PDF


def load_pdf(path: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Read the PDF and return all chunks across all pages."""
    reader = PdfReader(path)
    chunks: list[Chunk] = []
    for i, page in enumerate(reader.pages, start=1):
        # `or ""` guards against image-only pages where extract_text returns None.
        text = (page.extract_text() or "").strip()
        if text:
            chunks.extend(split_text(text, page=i, size=chunk_size, overlap=chunk_overlap))
    return chunks


def split_text(text: str, page: int, size: int, overlap: int) -> list[Chunk]:
    """
    Word-based sliding-window chunker.

    Example with size=10, overlap=2:
        words = [w1 w2 ... w20]
        chunk 1: w1..w10
        chunk 2: w9..w18   (step = size - overlap = 8)
        chunk 3: w17..w20

    Overlap exists so that an important sentence sitting on a boundary still
    appears INTACT in at least one chunk. Without it the sentence would be
    split in two and neither half would match a query well.

    We split on whitespace which is crude. A production system would split on
    sentences (NLTK / spaCy / a regex) for cleaner cuts.
    """
    words = text.split()
    if not words:
        return []

    out: list[Chunk] = []
    # `step` = how far the window advances each iteration.
    step = max(1, size - overlap)

    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size])
        if piece:
            out.append(Chunk(text=piece, page=page))
        # Stop once we've consumed the whole page; otherwise the final partial
        # window gets emitted as a shorter duplicate.
        if start + size >= len(words):
            break

    return out
