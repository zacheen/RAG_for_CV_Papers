"""Retrieve relevant chunks from ChromaDB given a query."""

import datetime

import chromadb

from src.config import TOP_K
from src.processing.embedder import get_collection


def _parse_result_date(metadata: dict) -> datetime.date | None:
    """Parse the best available date from chunk metadata."""
    hf_date = metadata.get("hf_date", "")
    if hf_date:
        try:
            return datetime.date.fromisoformat(hf_date)
        except ValueError:
            pass

    published = metadata.get("published", "")
    if published:
        try:
            return datetime.datetime.fromisoformat(
                published.replace("Z", "+00:00")
            ).date()
        except ValueError:
            pass

    return None


def retrieve(query: str, top_k: int = TOP_K,
             collection: chromadb.Collection | None = None,
             recent_days: int | None = None) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User query string.
        top_k: Number of results to return.
        collection: Optional ChromaDB collection.
        recent_days: If provided, filter results to papers published in the last N days.

    Returns:
        List of dicts with 'text', 'paper_id', 'title', 'distance' keys.
    """
    if collection is None:
        collection = get_collection()

    query_kwargs = {
        "query_texts": [query],
        "n_results": max(top_k * 5, 50) if recent_days is not None else top_k,
    }

    results = collection.query(**query_kwargs)

    retrieved = []
    threshold_date = None
    if recent_days is not None:
        threshold_date = datetime.date.today() - datetime.timedelta(days=recent_days)

    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None

            if threshold_date is not None:
                result_date = _parse_result_date(meta)
                if result_date is None or result_date < threshold_date:
                    continue

            retrieved.append({
                "text": doc,
                "paper_id": meta.get("paper_id", ""),
                "title": meta.get("title", ""),
                "arxiv_url": meta.get("arxiv_url", ""),
                "authors": meta.get("authors", ""),
                "published": meta.get("published", ""),
                "abstract": meta.get("abstract", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": distance,
            })

    return retrieved[:top_k]


def retrieve_recent_papers(
    recent_days: int,
    max_papers: int = TOP_K,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    """Return recent papers directly from ChromaDB metadata without vector search."""
    if collection is None:
        collection = get_collection()

    threshold_date = datetime.date.today() - datetime.timedelta(days=recent_days)
    papers: dict[str, dict] = {}
    batch_size = 500
    total_rows = collection.count()

    for offset in range(0, total_rows, batch_size):
        rows = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        documents = rows.get("documents", [])
        metadatas = rows.get("metadatas", [])

        for doc, meta in zip(documents, metadatas):
            result_date = _parse_result_date(meta or {})
            if result_date is None or result_date < threshold_date:
                continue

            paper_id = meta.get("paper_id", "")
            if not paper_id:
                continue

            abstract = (meta.get("abstract") or "").strip()
            chunk_index = meta.get("chunk_index", 0)
            text = abstract or (doc.strip() if chunk_index == 0 else "")
            if not text:
                continue

            existing = papers.get(paper_id)
            candidate = {
                "text": text,
                "paper_id": paper_id,
                "title": meta.get("title", ""),
                "arxiv_url": meta.get("arxiv_url", ""),
                "authors": meta.get("authors", ""),
                "published": meta.get("published", ""),
                "abstract": abstract,
                "chunk_index": chunk_index,
                "distance": 0.0,
            }

            if existing is None:
                papers[paper_id] = candidate
                continue

            # Prefer stored abstracts; otherwise keep the earliest chunk.
            if abstract and not existing.get("abstract"):
                papers[paper_id] = candidate
            elif not existing.get("abstract") and chunk_index < existing.get("chunk_index", 999999):
                papers[paper_id] = candidate

    results = list(papers.values())
    results.sort(
        key=lambda item: (
            _parse_result_date(item) or datetime.date.min,
            item.get("paper_id", ""),
        ),
        reverse=True,
    )
    return results[:max_papers]


def format_context(results: list[dict]) -> str:
    """Format retrieved results into a context string for the LLM.

    Args:
        results: List of retrieved chunk dicts.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        header = f"[{i}] {title}" if title else f"[{i}]"
        published = r.get("published", "Unknown")
        parts.append(
            f"{header}\n"
            f"Published: {published}\n"
            f"Authors: {r.get('authors', 'Unknown')}\n"
            f"Paper ID: {r.get('paper_id', 'Unknown')}\n"
            f"Retrieved content:\n{r['text']}"
        )
    return "\n\n".join(parts)
