"""Centralized configuration for the RAG pipeline."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# arXiv settings
ARXIV_CATEGORY = "cs.CV"
ARXIV_MAX_RESULTS = 800
ARXIV_SORT_BY = "submittedDate"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "arxiv_cv_papers"

# Ollama / LLM settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = 5

# RAG prompt template
RAG_SYSTEM_PROMPT = (
    "You are a research assistant specializing in computer vision. "
    "Answer the user's question based ONLY on the provided context from "
    "arXiv computer vision papers. If the context does not contain enough "
    "information, say so. Cite the paper titles when possible."
)

RAG_USER_TEMPLATE = (
    "Context from relevant papers:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}"
)
