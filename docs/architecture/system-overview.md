# System Architecture Overview

## High-Level Architecture

The system is a **Retrieval-Augmented Generation (RAG)** pipeline for computer vision research papers sourced from arXiv.

### Pipeline Stages

1. **Data Ingestion** — Search arXiv API for cs.CV papers, download PDFs (`src/ingestion/arxiv_downloader.py`)
2. **PDF Parsing** — Extract text from PDFs via PyMuPDF (`src/ingestion/pdf_parser.py`)
3. **Chunking** — Recursive semantic splitting into ~512-char chunks with overlap (`src/processing/chunker.py`)
4. **Embedding & Indexing** — Embed with all-MiniLM-L6-v2, store in ChromaDB (`src/processing/embedder.py`)
5. **Retrieval** — Cosine similarity search in ChromaDB for top-k chunks (`src/rag/retriever.py`)
6. **Generation** — LLaMA 3.2 via Ollama with RAG prompt template (`src/rag/generator.py`)
7. **UI** — Streamlit chat interface with streaming responses (`app.py`)

### Component Diagram

```
[arXiv API]
     |
     v
[arxiv_downloader.py] --> [pdf_parser.py] --> [chunker.py]
                                                    |
                                                    v
                                            [embedder.py]
                                          (all-MiniLM-L6-v2)
                                                    |
                                                    v
                                              [ChromaDB]
                                                    |
    [User Query via Streamlit] ---------------->    |
                                                    v
                                           [retriever.py]
                                             (top-k cosine)
                                                    |
                                                    v
                                           [generator.py]
                                         (LLaMA 3.2 / Ollama)
                                                    |
                                                    v
                                        [Streamed answer in UI]
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Data source | arXiv API (cs.CV category) |
| PDF parsing | PyMuPDF (fitz) |
| Chunking | Custom recursive splitter |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector store | ChromaDB (persistent, cosine distance) |
| LLM | LLaMA 3.2 via Ollama |
| UI | Streamlit |
| Deployment | Docker (single container: Ollama + Streamlit) |

## Configuration

All settings are centralized in `src/config.py`:
- Paths, arXiv parameters, chunk sizes, model names, prompt templates
- Ollama base URL is configurable via `OLLAMA_BASE_URL` env var

## Design Decisions

See [design-decisions.md](design-decisions.md) for the full decision log.
