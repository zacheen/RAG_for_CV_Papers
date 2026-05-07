"""Background worker that resolves citations and ingests cited papers.

Lives in its own module so :mod:`src.rag.download_state` can dispatch jobs
without importing the ingestion stack at module-load time (which would slow
down Streamlit startup).
"""

from __future__ import annotations

from src.ingestion.citation_resolver import (
    CitationResolverError,
    pick_references,
    resolve_references,
)
from src.ingestion.ingest_single import ingest_paper
from src.rag.download_state import (
    gate_check,
    mark_arxiv_in_flight,
    now_iso,
    record_result,
    release_arxiv,
)


def run_citation_download_job(
    source_paper_id: str, citation_indices: list[int]
) -> None:
    """Resolve a list of citation indices and ingest each cited paper.

    Always logs a result for every requested index — success, failure, or
    skipped — so the sidebar reflects what happened.
    """
    started_at = now_iso()

    try:
        references = resolve_references(source_paper_id)
    except CitationResolverError as exc:
        for idx in citation_indices:
            record_result(
                arxiv_id=f"{source_paper_id}#[{idx}]",
                status="failed",
                reason=f"resolve failed: {exc}",
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
        return

    picked = pick_references(references, citation_indices)
    for idx, entry in picked:
        cite_label = f"{source_paper_id}#[{idx}]"
        if entry is None:
            record_result(
                arxiv_id=cite_label,
                status="failed",
                reason=f"index {idx} out of range",
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
            continue

        arxiv_id = entry.get("arxiv_id")
        if not arxiv_id:
            title = entry.get("title", "") or "(unknown)"
            record_result(
                arxiv_id=cite_label,
                status="failed",
                reason=f"no arXiv id for '{title[:60]}'",
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
            continue

        # Re-run dedup gates right before doing real work.
        should, reason = gate_check(arxiv_id)
        if not should:
            record_result(
                arxiv_id=arxiv_id,
                status="skipped",
                reason=reason,
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
            continue

        if not mark_arxiv_in_flight(arxiv_id):
            record_result(
                arxiv_id=arxiv_id,
                status="skipped",
                reason="already in flight",
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
            continue

        try:
            result = ingest_paper(arxiv_id)
        finally:
            release_arxiv(arxiv_id)

        if result["status"] == "ok":
            record_result(
                arxiv_id=arxiv_id,
                status="ok",
                reason=f"indexed {result['chunks_indexed']} chunks",
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
        else:
            record_result(
                arxiv_id=arxiv_id,
                status="failed",
                reason=result.get("reason", "unknown"),
                source_paper_id=source_paper_id,
                started_at=started_at,
            )
