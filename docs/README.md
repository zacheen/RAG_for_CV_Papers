# CS 6120 NLP Final Project — RAG for Computer Vision Papers

A Retrieval-Augmented Generation system that ingests arXiv computer vision papers and enables semantic Q&A using LLaMA 3.2 via Ollama.

## Quick Links

| Resource | Location |
|----------|----------|
| System architecture | [architecture/system-overview.md](architecture/system-overview.md) |
| Design decisions log | [architecture/design-decisions.md](architecture/design-decisions.md) |
| Development setup & running | [development/setup.md](development/setup.md) |
| Course requirements | [project/requirements.md](project/requirements.md) |
| Lessons learned | [LESSONS.md](LESSONS.md) |
| Project description (PDF) | [../project-description.pdf](../project-description.pdf) |
| Configuration | [../src/config.py](../src/config.py) |

## Architecture at a Glance

```
arXiv API -> PyMuPDF -> Recursive Chunker -> all-MiniLM-L6-v2 -> ChromaDB -> LLaMA 3.2 -> Streamlit
```

For the full diagram and tech stack, see [system-overview.md](architecture/system-overview.md).

## Getting Started

See [development/setup.md](development/setup.md) for:
- Local development setup
- Docker deployment
- Running the ingestion pipeline
- Project file structure

## Key Source Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI with RAG |
| `scripts/ingest.py` | CLI to download, parse, chunk, and index papers |
| `src/config.py` | All configuration (paths, models, prompts) |
| `src/ingestion/arxiv_downloader.py` | arXiv API search and PDF download |
| `src/ingestion/pdf_parser.py` | PyMuPDF text extraction |
| `src/processing/chunker.py` | Recursive semantic text splitting |
| `src/processing/embedder.py` | Embedding + ChromaDB indexing |
| `src/rag/retriever.py` | Vector similarity retrieval |
| `src/rag/generator.py` | Ollama/LLaMA streaming generation |
| `Dockerfile` | Single container (Ollama + Streamlit) |
| `entrypoint.sh` | Container startup (Ollama + model pull + ingestion + app) |
