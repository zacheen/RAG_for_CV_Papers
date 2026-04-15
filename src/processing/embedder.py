"""Embed text chunks and store them in ChromaDB."""

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL


def get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """Get or create the ChromaDB collection with the configured embedding function.

    Args:
        client: Optional ChromaDB client. Creates one if not provided.

    Returns:
        ChromaDB Collection.
    """
    if client is None:
        client = get_chroma_client()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict], collection: chromadb.Collection | None = None) -> int:
    """Add document chunks to the ChromaDB collection.

    Args:
        chunks: List of dicts with 'text', 'paper_id', 'title', 'chunk_index'.
        collection: Optional ChromaDB collection. Creates default if not provided.

    Returns:
        Number of chunks indexed.
    """
    if collection is None:
        collection = get_collection()

    if not chunks:
        return 0

    ids = [f"{c['paper_id']}_chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "paper_id": c["paper_id"],
            "title": c["title"],
            "chunk_index": c["chunk_index"],
            "arxiv_url": c.get("arxiv_url", ""),
            "authors": c.get("authors", ""),
            "published": c.get("published", ""),
            "hf_date": c.get("hf_date", ""),
            "abstract": c.get("abstract", ""),
        }
        for c in chunks
    ]

    # ChromaDB upsert handles duplicates gracefully
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(ids)


def get_collection_stats(collection: chromadb.Collection | None = None) -> dict:
    """Get basic stats about the collection.

    Args:
        collection: Optional ChromaDB collection.

    Returns:
        Dict with 'total_chunks' and 'sample' keys.
    """
    if collection is None:
        collection = get_collection()

    count = collection.count()
    sample = collection.peek(limit=3) if count > 0 else {}
    return {"total_chunks": count, "sample": sample}
