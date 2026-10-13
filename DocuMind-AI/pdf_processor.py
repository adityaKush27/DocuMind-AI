"""PDF text extraction and page-aware chunking for DocuMind AI."""

from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader


class PDFProcessingError(Exception):
    """An expected, user-facing PDF validation or extraction problem."""


def clean_text(text: str) -> str:
    """Normalize whitespace without changing the words used for retrieval."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """Split text into readable passages with a small overlap between neighbors."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and greater than the overlap.")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer a sentence, line, or word boundary without making chunks tiny.
            minimum_boundary = start + int(chunk_size * 0.6)
            boundary = max(
                text.rfind(". ", minimum_boundary, end),
                text.rfind("? ", minimum_boundary, end),
                text.rfind("! ", minimum_boundary, end),
                text.rfind("\n", minimum_boundary, end),
                text.rfind(" ", minimum_boundary, end),
            )
            if boundary > start:
                end = boundary + (1 if text[boundary] in ".?!" else 0)

        passage = text[start:end].strip()
        if passage:
            chunks.append(passage)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


def extract_pdf(filename: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Extract text by page and produce page-aware 500-character passages."""
    if not pdf_bytes:
        raise PDFProcessingError("The file is empty.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise PDFProcessingError(
                    "This PDF is password-protected. Upload an unlocked copy."
                ) from exc
            if not decrypted:
                raise PDFProcessingError(
                    "This PDF is password-protected. Upload an unlocked copy."
                )
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "This file could not be opened as a PDF. Try exporting it again."
        ) from exc

    if not reader.pages:
        raise PDFProcessingError("This PDF does not contain any pages.")

    pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = clean_text(page.extract_text() or "")
        except Exception:
            page_text = ""
        pages.append({"page": page_number, "text": page_text})
        for passage in chunk_text(page_text):
            chunks.append(
                {
                    "text": passage,
                    "source": filename,
                    "page": page_number,
                }
            )

    if not chunks:
        raise PDFProcessingError(
            "No selectable text was found. This may be a scanned PDF; image-only PDFs need OCR."
        )

    return {
        "filename": filename,
        "page_count": len(reader.pages),
        "pages": pages,
        "chunks": chunks,
    }
