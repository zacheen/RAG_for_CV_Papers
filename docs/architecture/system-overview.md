# System Architecture Overview

## High-Level Architecture

The system is a **Retrieval-Augmented Generation (RAG)** pipeline for computer vision research papers sourced from arXiv, deployed in a **two-phase** architecture.

### Two-Phase Deployment

```
Phase 1: LOCAL                          Phase 2: GCP (GPU VM)
┌──────────────────────┐               ┌──────────────────────┐
│ arXiv API            │               │ ChromaDB (read-only) │
│   ↓                  │   upload      │   ↓                  │
│ PDF Download         │  data/ dir    │ Retriever            │
│   ↓                  │ ──────────>   │   ↓                  │
│ PyMuPDF Parser       │               │ LLaMA 3.2 (Ollama)  │
│   ↓                  │               │   ↓                  │
│ Recursive Chunker    │               │ Streamlit UI         │
│   ↓                  │               │                      │
│ Embedding (MiniLM)   │               │ Ports: 8501, 11434  │
│   ↓                  │               └──────────────────────┘
│ ChromaDB (write)     │
└──────────────────────┘
```

### Pipeline Stages

**Phase 1 — Ingestion (local, no GPU needed):**

1. **arXiv Search** — Query arXiv API for cs.CV papers, sorted by date (`src/ingestion/arxiv_downloader.py`)
2. **PDF Download** — Download PDFs with rate limiting (`src/ingestion/arxiv_downloader.py`)
3. **PDF Parsing** — Extract text via PyMuPDF (`src/ingestion/pdf_parser.py`)
4. **Chunking** — Recursive semantic splitting, ~512 chars with 64-char overlap (`src/processing/chunker.py`)
5. **Embedding & Indexing** — Embed with all-MiniLM-L6-v2, store in ChromaDB with metadata (`src/processing/embedder.py`)

**Phase 2 — Serving (GCP VM with GPU):**

6. **Retrieval** — Cosine similarity search in ChromaDB for top-k chunks (`src/rag/retriever.py`)
7. **Generation** — LLaMA 3.2 via Ollama with RAG prompt template (`src/rag/generator.py`)
8. **UI** — Streamlit chat with streaming responses and clickable citations (`app.py`)

### Metadata Flow (for clickable citations)

arXiv metadata is preserved end-to-end to enable clickable citations in the UI:

```
arxiv_downloader.py                chunker.py                  ChromaDB
  returns per paper:                attaches to each chunk:      stores per chunk:
  - id                              - paper_id                   - paper_id
  - title              ────>        - title              ────>  - title
  - authors                         - arxiv_url                  - arxiv_url
  - published                       - authors                    - authors
  - pdf_url                         - published                  - published
                                    - chunk_index                - chunk_index

                    retriever.py                    app.py
                      returns:                        displays:
                      - text                          - Clickable [Title](arxiv_url)
               ────>  - title                 ────>  - Author names
                      - arxiv_url                     - Passage preview (first 300 chars)
                      - authors                       - Similarity score
                      - distance
```

### Tech Stack

| Component | Technology | Runs on |
|-----------|-----------|---------|
| Data source | arXiv API (cs.CV category) | Local |
| PDF parsing | PyMuPDF (fitz) | Local |
| Chunking | Custom recursive splitter (512 chars, 64 overlap) | Local |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Local |
| Vector store | ChromaDB (persistent, cosine distance) | Local (build) → GCP (read) |
| LLM | LLaMA 3.2 via Ollama | GCP (T4 GPU) |
| UI | Streamlit | GCP |
| Deployment | Docker on GCP Compute Engine (n1-standard-8 + T4) | GCP |

## Configuration

All settings are centralized in `src/config.py`:
- Paths, arXiv parameters, chunk sizes, model names, prompt templates
- Ollama base URL is configurable via `OLLAMA_BASE_URL` env var
- Key defaults: `ARXIV_MAX_RESULTS=800`, `CHUNK_SIZE=512`, `TOP_K=5`

## Design Decisions

See [design-decisions.md](design-decisions.md) for the full decision log.
