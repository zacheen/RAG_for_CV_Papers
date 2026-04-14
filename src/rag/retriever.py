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
                "chunk_index": meta.get("chunk_index", 0),
                "distance": distance,
            })

    return retrieved[:top_k]


def format_context(results: list[dict]) -> str:
    """Format retrieved results into a context string for the LLM.

    Args:
        results: List of retrieved chunk dicts.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['title']}" if r["title"] else f"[{i}]"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n".join(parts)
