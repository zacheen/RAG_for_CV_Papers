"""Entry point for daily cron job to ingest today's HF Daily Papers."""

import sys
from datetime import datetime
from pathlib import Path

# Allow running as `python scripts/get_today_trend.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.hf_downloader import fetch_daily_cv_papers
from src.ingestion.arxiv_downloader import download_pdf
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import get_collection, index_chunks, get_collection_stats
from src.config import PDF_DIR

def run_today_trend():
    """Fetch, download, parse and index today's HF CV papers."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching HF Daily CV Papers for {today_str}...")
    
    papers = fetch_daily_cv_papers(today_str, max_papers=2)
    if not papers:
        print("No CV papers found today.")
        return

    collection = get_collection()
    total_chunks = 0
    
    print(f"Found {len(papers)} CV papers. Processing...")
    
    for paper in papers:
        paper_id = paper["id"]
        print(f"Processing paper: {paper['title']} ({paper_id})")
        
        pdf_path = download_pdf(paper, PDF_DIR)
        if not pdf_path:
            continue
            
        try:
            parsed = parse_pdf(pdf_path)
            text = parsed["text"]

            if len(text.strip()) < 100:
                print(f"  Skipping {paper_id}: too little text extracted")
                continue

            authors = ", ".join(paper.get("authors", []))
            
            chunks = chunk_document(
                text, 
                paper_id=paper_id, 
                title=paper.get("title", ""),
                arxiv_url=paper.get("pdf_url", f"https://arxiv.org/abs/{paper_id}").replace("/pdf/", "/abs/"), 
                authors=authors, 
                published=paper.get("published", ""),
                hf_date=today_str,  # Crucial for Recent Filtering!
                abstract=paper.get("summary", ""),
            )
            
            indexed = index_chunks(chunks, collection=collection)
            total_chunks += indexed
            print(f"  Indexed {indexed} chunks for {paper_id}")
            
        except Exception as e:
            print(f"  Error processing {paper_id}: {e}")
            continue

    stats = get_collection_stats(collection)
    print("\nDaily Ingestion Complete!")
    print(f"  New chunks indexed today: {total_chunks}")
    print(f"  Total chunks in collection: {stats['total_chunks']}")

if __name__ == "__main__":
    run_today_trend()
