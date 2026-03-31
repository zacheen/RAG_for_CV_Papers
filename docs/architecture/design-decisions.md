# Design Decisions

This document tracks all design decisions for the project.

## Resolved Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Project track | RAG LLM | Per course project template |
| 2 | Data source | arXiv CV papers (cs.CV) | User decision — open access, large volume |
| 3 | Data acquisition | arXiv API | Official API, clean, respects ToS |
| 4 | PDF parsing | PyMuPDF (fitz) | Lightweight, no external service, sufficient for arXiv PDFs |
| 5 | Chunking strategy | Recursive/semantic (512 chars, 64 overlap) | Best balance of semantic coherence and consistent chunk sizes |
| 6 | Embedding model | all-MiniLM-L6-v2 | Lightweight, fast, well-integrated with ChromaDB |
| 7 | Vector store | ChromaDB | From professor's template code |
| 8 | LLM | LLaMA 3.2 via Ollama | User choice + template code uses `ollama` library |
| 9 | Deployment | Docker + Streamlit | From professor's template code |
| 10 | Two-phase split | Local ingestion + GCP serving | Saves GPU costs; ingestion doesn't need GPU |
| 11 | Clickable citations | arXiv metadata (url, authors, published) stored in ChromaDB | Course requirement: clickable citation to data source |
| 12 | Data scale | 800 papers → ~16k chunks | Course requirement: database >= 10k entries |

## Open Decisions

None — all decisions resolved.
