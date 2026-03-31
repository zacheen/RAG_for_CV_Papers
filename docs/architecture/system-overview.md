# System Architecture Overview

## High-Level Architecture

The system is a **Retrieval-Augmented Generation (RAG)** pipeline for computer vision research papers sourced from arXiv.

### Pipeline Stages

1. **Data Ingestion** — Download and parse arXiv CV papers (PDFs or LaTeX source)
2. **Preprocessing** — Extract text, clean, and normalize academic content (equations, figures, references)
3. **Chunking** — Split papers into semantically meaningful chunks (by section, paragraph, or sliding window)
4. **Embedding** — Generate vector embeddings for each chunk using a sentence-transformer model
5. **Indexing** — Store embeddings in a vector database for efficient similarity search
6. **Retrieval** — Given a user query, retrieve the top-k most relevant chunks
7. **Generation** — Feed retrieved context + query to an open-source LLM to generate an answer

### Component Diagram

```
[arXiv API / Bulk Download]
        |
        v
[PDF/LaTeX Parser] --> [Text Preprocessor] --> [Chunker]
                                                   |
                                                   v
                                            [Embedding Model]
                                                   |
                                                   v
                                            [Vector Store]
                                                   |
        [User Query] --> [Query Encoder] --------->|
                                                   |
                                                   v
                                          [Retriever (top-k)]
                                                   |
                                                   v
                                    [Prompt Builder (query + context)]
                                                   |
                                                   v
                                          [Open-Source LLM]
                                                   |
                                                   v
                                        [Generated Answer]
```

## Key Design Decisions (TBD)

See [design-decisions.md](design-decisions.md) for open choices that need to be resolved before implementation.
