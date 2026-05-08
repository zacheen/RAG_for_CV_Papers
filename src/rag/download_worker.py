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
    search_arxiv_by_title,
)
from src.ingestion.ingest_single import ingest_paper, ingest_pdf_from_url
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
            cite_label = f"{source_paper_id}#[{idx}]"
            record_result(
                arxiv_id=cite_label,
                status="failed",
                reason=f"resolve failed: {exc}",
                source_paper_id=source_paper_id,
                started_at=started_at,
                cite_label=cite_label,
            )
        return

    picked = pick_references(references, citation_indices)
    for idx, entry in picked:
        cite_label = f"{source_paper_id}#[{idx}]"

        # Flip the pre-populated pending row to running so the user sees
        # progress on the active citation. record_result upserts by
        # cite_label, so this updates in place.
        record_result(
            arxiv_id=cite_label,
            status="running",
            reason="resolving",
            source_paper_id=source_paper_id,
            started_at=started_at,
            cite_label=cite_label,
        )

        if entry is None:
            record_result(
                arxiv_id=cite_label,
                status="failed",
                reason=f"index {idx} out of range",
                source_paper_id=source_paper_id,
                started_at=started_at,
                cite_label=cite_label,
            )
            continue

        arxiv_id = entry.get("arxiv_id")
        title = entry.get("title", "") or ""
        title_search_diag: dict = {}
        if not arxiv_id and title:
            # Semantic Scholar often omits externalIds.ArXiv even for papers
            # that are on arXiv. Fall back to a title-based arXiv search.
            title_search_diag = search_arxiv_by_title(title)
            arxiv_id = title_search_diag.get("arxiv_id")
        if not arxiv_id:
            # OA fallback: SS often supplies an open-access PDF URL even when
            # the paper isn't on arXiv. Try that before giving up so journal
            # / bioRxiv / publisher-OA references still get indexed.
            pdf_url = entry.get("pdf_url")
            ss_id = entry.get("paper_id_ss")
            if pdf_url and ss_id:
                synthetic_id = f"ss_{ss_id}"
                should, reason = gate_check(synthetic_id)
                if not should:
                    record_result(
                        arxiv_id=synthetic_id,
                        status="skipped",
                        reason=reason,
                        source_paper_id=source_paper_id,
                        started_at=started_at,
                        cite_label=cite_label,
                    )
                    continue
                if not mark_arxiv_in_flight(synthetic_id):
                    record_result(
                        arxiv_id=synthetic_id,
                        status="skipped",
                        reason="already in flight",
                        source_paper_id=source_paper_id,
                        started_at=started_at,
                        cite_label=cite_label,
                    )
                    continue
                record_result(
                    arxiv_id=synthetic_id,
                    status="running",
                    reason="downloading via OA fallback",
                    source_paper_id=source_paper_id,
                    started_at=started_at,
                    cite_label=cite_label,
                )
                try:
                    oa_result = ingest_pdf_from_url(
                        pdf_url, paper_id=synthetic_id, title=title
                    )
                finally:
                    release_arxiv(synthetic_id)
                if oa_result["status"] == "ok":
                    record_result(
                        arxiv_id=synthetic_id,
                        status="ok",
                        reason=f"indexed {oa_result['chunks_indexed']} chunks (OA fallback)",
                        source_paper_id=source_paper_id,
                        started_at=started_at,
                        cite_label=cite_label,
                    )
                else:
                    record_result(
                        arxiv_id=synthetic_id,
                        status="failed",
                        reason=f"OA fallback: {oa_result.get('reason', 'unknown')}",
                        source_paper_id=source_paper_id,
                        started_at=started_at,
                        cite_label=cite_label,
                    )
                continue

            reason_parts = [f"no arXiv id for '{title or '(unknown)'}'"]
            if title_search_diag:
                if title_search_diag.get("error"):
                    reason_parts.append(
                        f"title search error: {title_search_diag['error']}"
                    )
                else:
                    reason_parts.append(
                        "title search: best match "
                        f"'{title_search_diag.get('best_candidate_title', '') or '(no candidates)'}' "
                        f"score={title_search_diag.get('best_score', 0.0):.2f} "
                        f"(inspected {title_search_diag.get('candidates_inspected', 0)})"
                    )
            record_result(
                arxiv_id=cite_label,
                status="failed",
                reason=" | ".join(reason_parts),
                source_paper_id=source_paper_id,
                started_at=started_at,
                cite_label=cite_label,
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
                cite_label=cite_label,
            )
            continue

        if not mark_arxiv_in_flight(arxiv_id):
            record_result(
                arxiv_id=arxiv_id,
                status="skipped",
                reason="already in flight",
                source_paper_id=source_paper_id,
                started_at=started_at,
                cite_label=cite_label,
            )
            continue

        record_result(
            arxiv_id=arxiv_id,
            status="running",
            reason="downloading from arXiv",
            source_paper_id=source_paper_id,
            started_at=started_at,
            cite_label=cite_label,
        )
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
                cite_label=cite_label,
            )
        else:
            record_result(
                arxiv_id=arxiv_id,
                status="failed",
                reason=result.get("reason", "unknown"),
                source_paper_id=source_paper_id,
                started_at=started_at,
                cite_label=cite_label,
            )
