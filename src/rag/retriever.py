"""Retrieve relevant chunks from ChromaDB given a query."""

import datetime
import chromadb

from src.config import TOP_K
from src.processing.embedder import get_collection


def retrieve(query: str, top_k: int = TOP_K,
             collection: chromadb.Collection | None = None,
             recent_days: int | None = None) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User query string.
        top_k: Number of results to return.
        collection: Optional ChromaDB collection.
        recent_days: If provided, filter results to papers indexed in the last N days.

    Returns:
        List of dicts with 'text', 'paper_id', 'title', 'distance' keys.
    """
    if collection is None:
        collection = get_collection()

    query_kwargs = {
        "query_texts": [query],
        "n_results": top_k,
    }

    if recent_days is not None:
        threshold_date = (datetime.datetime.now() - datetime.timedelta(days=recent_days)).strftime("%Y-%m-%d")
        query_kwargs["where"] = {"hf_date": {"$gte": threshold_date}}

    results = collection.query(**query_kwargs)

    retrieved = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
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

    return retrieved


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
