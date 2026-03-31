# CS 6120 NLP Final Project — RAG for Computer Vision Papers

A Retrieval-Augmented Generation system that ingests arXiv computer vision papers and enables semantic Q&A using an open-source LLM.

## Quick Links

| Resource | Location |
|----------|----------|
| Project requirements | [docs/project/requirements.md](project/requirements.md) |
| System architecture | [docs/architecture/system-overview.md](architecture/system-overview.md) |
| Open design decisions | [docs/architecture/design-decisions.md](architecture/design-decisions.md) |
| Development setup | [docs/development/setup.md](development/setup.md) |
| Lessons learned | [LESSONS.md](LESSONS.md) |
| Project description (PDF) | [project-description.pdf](../project-description.pdf) |

## Architecture at a Glance

The system follows a standard RAG pipeline:

**Ingestion** (arXiv download) -> **Parsing** (PDF/LaTeX) -> **Chunking** -> **Embedding** -> **Vector Store** -> **Retrieval** -> **LLM Generation**

For details, see [system-overview.md](architecture/system-overview.md).

## Open Design Decisions

Six design choices remain open before implementation can begin. See [design-decisions.md](architecture/design-decisions.md) for the full options matrix covering:

1. PDF parsing strategy
2. Chunking strategy
3. Embedding model
4. Vector store
5. Open-source LLM selection
6. Deployment / serving approach

## Project Structure (Planned)

```
final_project/
  CLAUDE.md                 # Agent guidance
  project-description.pdf   # Course assignment spec
  docs/
    README.md               # This file (reference index)
    LESSONS.md              # Sessions and lessons learned
    architecture/           # System design docs
    development/            # Setup, running, testing guides
    project/                # Requirements and course deliverables
  src/                      # (TBD) Application source code
  data/                     # (TBD) Data scripts and sample data
  Dockerfile                # (TBD) Container definition
  requirements.txt          # (TBD) Python dependencies
```
