# CS 6120 NLP Final Project — RAG for Computer Vision Papers

A Retrieval-Augmented Generation system that ingests arXiv computer vision papers and enables semantic Q&A using LLaMA 3.2 via Ollama.

## Quick Links

| Resource | Location |
|----------|----------|
| System architecture | [architecture/system-overview.md](architecture/system-overview.md) |
| Design decisions log | [architecture/design-decisions.md](architecture/design-decisions.md) |
| Local ingestion & setup | [development/setup.md](development/setup.md) |
| GCP deployment guide | [development/gcp-deployment.md](development/gcp-deployment.md) |
| Course requirements & compliance | [project/requirements.md](project/requirements.md) |
| Lessons learned | [LESSONS.md](LESSONS.md) |
| Configuration | [../src/config.py](../src/config.py) |
| Project description (PDF) | [../project-description.pdf](../project-description.pdf) |

## Architecture at a Glance

Two-phase deployment:

```
Phase 1 (Local):   arXiv API → PyMuPDF → Chunker → MiniLM Embedding → ChromaDB
                                                                          │
                                                                    upload data/
                                                                          │
Phase 2 (GCP):                                          ChromaDB → Retriever → LLaMA 3.2 → Streamlit UI
```

For the full diagram, metadata flow, and tech stack, see [system-overview.md](architecture/system-overview.md).

## Getting Started

1. **Run ingestion locally:** `python scripts/ingest.py --max-papers 800` (see [setup.md](development/setup.md))
2. **Upload to GCP:** `gcloud compute scp --recurse ./data <VM>:~/cv-paper-rag/data` (see [gcp-deployment.md](development/gcp-deployment.md))
3. **Start the app on GCP:** Docker container runs Ollama + Streamlit automatically

## Key Source Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI with RAG and clickable citations |
| `scripts/ingest.py` | CLI to download, parse, chunk, and index papers (run locally) |
| `src/config.py` | All configuration (paths, models, prompts, scale) |
| `src/ingestion/arxiv_downloader.py` | arXiv API search and PDF download |
| `src/ingestion/pdf_parser.py` | PyMuPDF text extraction |
| `src/processing/chunker.py` | Recursive semantic text splitting |
| `src/processing/embedder.py` | Embedding + ChromaDB indexing (with citation metadata) |
| `src/rag/retriever.py` | Vector similarity retrieval (returns citation metadata) |
| `src/rag/generator.py` | Ollama/LLaMA streaming generation |
| `Dockerfile` | Single container (Ollama + Streamlit) |
| `entrypoint.sh` | Container startup (Ollama + model pull + Streamlit) |

## Technical Requirements Compliance

| Requirement | Status |
|-------------|--------|
| Front end (Streamlit) | Done |
| Database >= 10k entries | Done (800 papers → ~16k chunks) |
| LLM entirely local on GCP | Done (LLaMA 3.2 via Ollama, T4 GPU) |
| Clickable citations | Done (arXiv links + passage preview) |

Details: [project/requirements.md](project/requirements.md)
