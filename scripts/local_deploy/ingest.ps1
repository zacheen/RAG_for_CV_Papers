# Local - Crawl arXiv cs.CV papers and build the local ChromaDB.
# Activate your conda/venv environment first, then run from project root.

python data/ingest.py --max-papers 800 # --query TOPIC
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
