"""Recursive text chunking for RAG pipeline."""

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP,
                    separators: list[str] | None = None) -> list[str]:
    """Recursively split text into chunks respecting semantic boundaries.

    Tries each separator in order, splitting on the first one that produces
    chunks under the size limit. Falls back to character-level splitting.

    Args:
        text: Input text to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separators to try.

    Returns:
        List of text chunks.
    """
    if separators is None:
        separators = list(SEPARATORS)

    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator
    for i, sep in enumerate(separators):
        if sep and sep in text:
            parts = text.split(sep)
            remaining_separators = separators[i:]
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part

                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    # If this single part exceeds chunk_size, split it further
                    if len(part) > chunk_size:
                        sub_chunks = recursive_split(
                            part, chunk_size, chunk_overlap, remaining_separators[1:]
                        )
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part

            if current and current.strip():
                chunks.append(current.strip())

            # Apply overlap
            if chunk_overlap > 0 and len(chunks) > 1:
                chunks = _apply_overlap(chunks, chunk_overlap)

            return [c for c in chunks if c.strip()]

    # Last resort: character-level split
    chunks = []
    for start in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap between consecutive chunks.

    Args:
        chunks: List of text chunks.
        overlap: Number of characters to overlap.

    Returns:
        Chunks with overlap applied.
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(prev_tail + " " + chunks[i])
    return result


def chunk_document(text: str, paper_id: str = "",
                   title: str = "", arxiv_url: str = "",
                   authors: str = "", published: str = "") -> list[dict]:
    """Chunk a document and attach metadata to each chunk.

    Args:
        text: Full document text.
        paper_id: arXiv paper ID.
        title: Paper title.
        arxiv_url: arXiv abstract URL for citation linking.
        authors: Comma-separated author names.
        published: Publication date string.

    Returns:
        List of dicts with text and metadata keys.
    """
    raw_chunks = recursive_split(text)
    return [
        {
            "text": chunk,
            "paper_id": paper_id,
            "title": title,
            "chunk_index": i,
            "arxiv_url": arxiv_url,
            "authors": authors,
            "published": published,
        }
        for i, chunk in enumerate(raw_chunks)
    ]
