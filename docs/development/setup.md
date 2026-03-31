# Development Setup

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Ollama installed locally (for local development without Docker)

## Local Development

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull LLaMA

```bash
ollama serve &
ollama pull llama3.2
```

### 3. Run ingestion

```bash
# Download and index 50 papers (default)
python scripts/ingest.py --max-papers 50

# With a search query filter
python scripts/ingest.py --query "object detection" --max-papers 100

# Process already-downloaded PDFs only
python scripts/ingest.py --skip-download
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Docker Deployment

### Build and run

```bash
docker build -t cv-paper-rag .
docker run -p 8501:8501 -p 11434:11434 -v ./data:/root/data cv-paper-rag
```

The container will:
1. Start the Ollama server
2. Pull LLaMA 3.2 (first run only)
3. Run ingestion if no indexed data exists
4. Start the Streamlit app on port 8501

## Project Structure

```
final_project/
  app.py                      # Streamlit RAG chatbot UI
  requirements.txt             # Python dependencies
  Dockerfile                   # Single container (Ollama + app)
  entrypoint.sh                # Container startup script
  .gitignore
  CLAUDE.md                    # Agent guidance
  project-description.pdf      # Course assignment spec
  scripts/
    ingest.py                  # CLI ingestion pipeline
  src/
    config.py                  # Centralized configuration
    ingestion/
      arxiv_downloader.py      # arXiv API search + PDF download
      pdf_parser.py            # PyMuPDF text extraction
    processing/
      chunker.py               # Recursive text chunking
      embedder.py              # Embedding + ChromaDB indexing
    rag/
      retriever.py             # ChromaDB query retrieval
      generator.py             # Ollama/LLaMA answer generation
  data/                        # (gitignored) Downloaded PDFs + ChromaDB
  docs/                        # Project documentation
```

## Testing

Run individual modules:

```bash
# Test arXiv search (returns metadata, no download)
python -c "from src.ingestion.arxiv_downloader import search_arxiv; print(search_arxiv(max_results=3))"

# Test PDF parsing
python -c "from src.ingestion.pdf_parser import parse_pdf; from pathlib import Path; print(parse_pdf(Path('data/pdfs/SOME_ID.pdf'))['text'][:500])"

# Check ChromaDB collection stats
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
```
