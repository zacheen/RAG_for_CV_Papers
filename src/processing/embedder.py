"""Embed text chunks and store them in ChromaDB."""

import sqlite3
import threading

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL


# Set the first time get_collection() successfully builds the
# SentenceTransformer-backed collection. UI code reads this to decide whether
# to show "Loading embedding model..." vs "Retrieving..." in status updates.
# Process-lifetime; survives Streamlit reruns because module import is cached.
_embedder_loaded = threading.Event()


def is_embedder_loaded() -> bool:
    """True iff the SentenceTransformer-backed collection has been built at
    least once in this process (i.e. ``get_collection()`` succeeded)."""
    return _embedder_loaded.is_set()


def get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """Get or create the ChromaDB collection with the configured embedding function.

    Use this for operations that need to embed text on the fly:
    ``query(query_texts=...)`` and ``upsert(documents=...)``. Building this
    collection loads SentenceTransformer (~17s cold), so prefer
    :func:`get_collection_lite` when you only need metadata.

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

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    _embedder_loaded.set()
    return collection


def get_collection_lite(
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    """Get or create the collection WITHOUT loading any embedding model.

    Use this for metadata-only operations: ``count()``, ``get(where=...)``,
    ``get(ids=[...])``. Calling ``query(query_texts=...)`` or
    ``upsert(documents=...)`` on the returned collection will fall back to
    ChromaDB's default embedder, which is not what we index with — so don't.

    Cold cost ~0.4s vs ~17s for :func:`get_collection`.

    Args:
        client: Optional ChromaDB client. Creates one if not provided.

    Returns:
        ChromaDB Collection without a project-specific embedding function.
    """
    if client is None:
        client = get_chroma_client()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
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


def get_chunk_count_fast() -> int:
    """Return the indexed-chunk count by querying ChromaDB's underlying
    SQLite directly, bypassing ChromaDB's collection layer.

    Uses ``MAX(rowid)`` instead of ``COUNT(*)`` because:

    - ``COUNT(*)`` walks the full ``embeddings`` b-tree — O(N) page reads.
      Measured at ~62s cold for 303k rows on a 2.3GB DB after reboot.
    - ``MAX(rowid)`` follows the rightmost path of the b-tree — O(log N).
      Should be sub-second even on cold OS cache.

    Correctness assumes:
      1. rowids are sequential starting at 1 (default SQLite INSERT
         behavior — ChromaDB doesn't override this).
      2. No row has ever been deleted from the ``embeddings`` table.

    Both hold in this project today: ingestion only ever appends new
    chunks. If chunk removal is added later, switch to a cached-on-disk
    counter file written during ingestion (e.g. ``data/chroma_db/
    chunk_count.txt``) to keep first-paint fast.

    Returns 0 if the database file doesn't exist yet (before first ingest).
    """
    db_path = CHROMA_DIR / "chroma.sqlite3"
    if not db_path.exists():
        return 0
    # Read-only mode keeps us safe from any writer (the Streamlit app may
    # have ChromaDB clients open in parallel).
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        row = conn.execute("SELECT MAX(rowid) FROM embeddings").fetchone()
        return row[0] or 0
