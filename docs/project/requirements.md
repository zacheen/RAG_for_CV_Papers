# Project Requirements

## Course Requirements (from CS 6120 Final Project Template)

### Deliverables

1. **PDF Report** — Following the template structure (Introduction, Background, Approach, Data, Results, Conclusions)
2. **Git Repository** — With README and reproducible setup
3. **Endpoint / Demo** — GCP endpoint for live demonstration
4. **Gradescope Submission** — PDF, endpoint link, and GitHub URL

### Report Sections Required

1. Objectives and Introduction
2. Background and Related Work (data origin, attributes, access instructions)
3. Approach and Implementation (design decisions, algorithms, README link)
4. Data and Data Analysis (dataset citation, distribution analysis)
5. Results and Evaluation (metrics, confidence, subjective examples)
6. Conclusions (learnings, applications, future work)

---

## Technical Requirements — Compliance Matrix

| # | Requirement | How We Meet It | Status |
|---|-------------|---------------|--------|
| 1 | **Front end** (e.g., Streamlit) | `app.py` — Streamlit chat UI with streaming responses | Done |
| 2 | **Database >= 10k entries** | 800 arXiv papers x ~20 chunks/paper = ~16k ChromaDB entries | Done (config) |
| 3 | **LLM entirely local** (on GCP or metal) | LLaMA 3.2 via Ollama on GCP Compute Engine VM (T4 GPU) | Done (architecture) |
| 4 | **Clickable citation** to data source (article + passage) | Each chunk stores `arxiv_url`, `authors`, `published` in ChromaDB; UI renders clickable `[Title](arxiv_url)` links with passage preview | Done |

### Requirement Details

#### 1. Front End

- **Technology:** Streamlit
- **File:** `app.py`
- **Features:** Chat interface, streaming responses, sidebar settings, collection stats, expandable source citations

#### 2. Database >= 10k Entries

- **Technology:** ChromaDB (persistent, cosine distance)
- **Scale:** 800 papers, each yielding ~20 chunks = ~16,000 entries
- **Config:** `src/config.py` → `ARXIV_MAX_RESULTS = 800`
- **Verification:** `python -c "from src.processing.embedder import get_collection_stats; print(get_collection_stats())"`

#### 3. LLM Entirely Local

- **Model:** LLaMA 3.2 (via Ollama)
- **Infrastructure:** GCP Compute Engine VM (`n1-standard-8` + NVIDIA T4 GPU)
- **No external API calls:** Ollama runs locally inside the Docker container
- **File:** `src/rag/generator.py` calls `ollama.chat()` which connects to localhost:11434

#### 4. Clickable Citations

- **Metadata stored per chunk:** `arxiv_url`, `title`, `authors`, `published`, `chunk_index`
- **UI display:** Clickable arXiv link, author names, passage preview (first 300 chars), similarity score
- **Files involved:** `embedder.py` (stores metadata) → `retriever.py` (returns metadata) → `app.py` (renders links)

---

## Data Source

- **Source:** arXiv API — Computer Vision and Pattern Recognition (cs.CV)
- **URL:** https://arxiv.org/list/cs.CV/recent
- **API docs:** https://info.arxiv.org/help/api/index.html
- **Selection:** Most recent papers, sorted by submission date descending
- **License:** arXiv papers are open access
