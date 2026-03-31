"""Retrieve relevant chunks from ChromaDB given a query."""

import chromadb

from src.config import TOP_K
from src.processing.embedder import get_collection


def retrieve(query: str, top_k: int = TOP_K,
             collection: chromadb.Collection | None = None) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User query string.
        top_k: Number of results to return.
        collection: Optional ChromaDB collection.

    Returns:
        List of dicts with 'text', 'paper_id', 'title', 'distance' keys.
    """
    if collection is None:
        collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

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
