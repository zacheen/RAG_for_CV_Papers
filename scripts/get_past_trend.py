"""Entry point to ingest past year of HF Daily Papers."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running as `python scripts/get_past_trend.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.hf_downloader import fetch_daily_cv_papers
from src.ingestion.arxiv_downloader import download_pdf
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import get_collection, index_chunks, get_collection_stats
from src.config import PDF_DIR

def run_past_trend(days_back: int = 365):
    """Fetch, download, parse and index HF CV papers for the past N days."""
    today = datetime.now()
    collection = get_collection()
    
    total_papers = 0
    total_chunks = 0
    
    for i in range(days_back):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"\n--- Fetching HF Daily CV Papers for {date_str} ---")
        papers = fetch_daily_cv_papers(date_str, max_papers=2)
        
        if not papers:
            print("  No CV papers found.")
            continue
            
        print(f"  Found {len(papers)} CV papers.")
        
        for paper in papers:
            paper_id = paper["id"]
            pdf_path = download_pdf(paper, PDF_DIR)
            if not pdf_path:
                print(f"  Failed to download {paper_id}")
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
                    hf_date=date_str  # Crucial for Recent Filtering!
                )
                
                indexed = index_chunks(chunks, collection=collection)
                total_chunks += indexed
                total_papers += 1
                print(f"  Indexed {indexed} chunks for {paper_id}")
                
            except Exception as e:
                print(f"  Error processing {paper_id}: {e}")
                continue

    stats = get_collection_stats(collection)
    print(f"\nPast Trends Ingestion Complete! ({days_back} days)")
    print(f"  Total papers processed: {total_papers}")
    print(f"  New chunks indexed: {total_chunks}")
    print(f"  Total chunks in collection: {stats['total_chunks']}")

if __name__ == "__main__":
    run_past_trend(days_back=365)
