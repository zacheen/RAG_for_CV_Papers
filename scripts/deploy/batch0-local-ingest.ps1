# Batch 0 [PS] - Local Ingestion
# Run from project root: D:\dont_move\Northeastern_University\CS6120\final_project

pip install PyMuPDF chromadb sentence-transformers
python scripts/ingest.py --max-papers 800 # --query TOPIC
python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"
