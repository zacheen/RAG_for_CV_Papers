"""Compare ChromaDB count() vs raw SQLite count to isolate the bottleneck."""
import sqlite3
import time
from pathlib import Path

CHROMA_DIR = Path("data/chroma_db")
SQLITE_PATH = CHROMA_DIR / "chroma.sqlite3"

print(f"chroma.sqlite3 size: {SQLITE_PATH.stat().st_size / (1024**3):.2f} GB")

# Connect read-only to avoid any write contention with Streamlit if it's running
t0 = time.perf_counter()
conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
print(f"{time.perf_counter()-t0:6.2f}s  sqlite3.connect (read-only)")

# What tables exist?
t0 = time.perf_counter()
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print(f"{time.perf_counter()-t0:6.2f}s  list tables")
print(f"        tables: {tables}")

# ChromaDB stores embeddings in `embeddings` table (or `embedding_metadata` etc.)
# Let's count rows in each candidate table.
candidates = [t for t in tables if "embed" in t.lower() or t.lower() in ("collections", "segments")]
for table in candidates:
    t0 = time.perf_counter()
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{time.perf_counter()-t0:6.2f}s  SELECT COUNT(*) FROM {table} = {count:,}")

conn.close()
