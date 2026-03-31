# Design Decisions

This document tracks all design decisions for the project.

## Resolved Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Project track | RAG LLM | Per course project template |
| 2 | Data source | arXiv CV papers (cs.CV) | User decision |
| 3 | Data acquisition | arXiv API | Official API, clean, respects ToS |
| 4 | PDF parsing | PyMuPDF (fitz) | Lightweight, no external service, sufficient for arXiv PDFs |
| 5 | Chunking strategy | Recursive/semantic | Best balance of semantic coherence and consistent chunk sizes for RAG |
| 6 | Embedding model | all-MiniLM-L6-v2 | Lightweight, fast, well-integrated with ChromaDB sentence-transformers |
| 7 | Vector store | ChromaDB | From professor's template code (`requirements.txt`) |
| 8 | LLM | LLaMA via Ollama | User choice + template code uses `ollama` library |
| 9 | Deployment | Docker + Streamlit | From professor's template code (`app.py`) |

## Open Decisions

None — all decisions resolved. Ready for implementation upon user authorization.
