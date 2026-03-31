"""Extract text from PDF files using PyMuPDF (fitz)."""

from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def extract_metadata_from_pdf(pdf_path: Path) -> dict:
    """Extract basic metadata from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with metadata fields (title, author, etc.).
    """
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}
    page_count = len(doc)
    doc.close()
    metadata["page_count"] = page_count
    return metadata


def parse_pdf(pdf_path: Path) -> dict:
    """Parse a PDF into text and metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with 'text', 'metadata', and 'source' keys.
    """
    text = extract_text_from_pdf(pdf_path)
    metadata = extract_metadata_from_pdf(pdf_path)
    return {
        "text": text,
        "metadata": metadata,
        "source": str(pdf_path),
    }
