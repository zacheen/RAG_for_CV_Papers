"""Gemini Native Function Calling tools for the RAG chatbot.

These three functions are passed to ``GenerateContentConfig(tools=[...])``
during the pre-RAG pass. Gemini's Automatic Function Calling introspects
their signatures and docstrings to decide when to invoke them.

State is held at module level so the tools stay pure-Python (no Streamlit
imports). ``app.py`` syncs ``st.session_state["time_range"]`` to and from the
:data:`time_range_state` object across each chat turn.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from src.rag.download_state import enqueue_citation_download


@dataclass
class TimeRangeState:
    """Module-level mirror of the active publication-date filter."""

    start_date: str | None = None
    end_date: str | None = None

    def clear(self) -> None:
        self.start_date = None
        self.end_date = None

    def set(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date

    def to_dict(self) -> dict[str, str | None]:
        return {"start_date": self.start_date, "end_date": self.end_date}


time_range_state = TimeRangeState()


def _validate_iso_date(value: str, field_name: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an ISO date YYYY-MM-DD, got {value!r}"
        ) from exc


def set_time_range(start_date: str, end_date: str) -> str:
    """Set the publication-date range used to filter retrieved papers.

    Call this whenever the user wants to constrain results to papers
    published within a specific window. The LLM is responsible for
    converting natural-language phrases ("last month", "March 2025",
    "since 2024") into absolute ISO dates using today's date provided in
    the system prompt. Both dates are inclusive.

    Args:
        start_date: ISO date string YYYY-MM-DD (inclusive).
        end_date: ISO date string YYYY-MM-DD (inclusive). If the user did
            not specify an end, pass today's date.

    Returns:
        Human-readable confirmation string.
    """
    try:
        start = _validate_iso_date(start_date, "start_date")
        end = _validate_iso_date(end_date, "end_date")
    except ValueError as exc:
        return f"Error: {exc}"

    if start > end:
        return (
            f"Error: start_date {start_date} is after end_date {end_date}; "
            "no change applied."
        )

    time_range_state.set(start.isoformat(), end.isoformat())
    return f"Time range set to {start.isoformat()} -> {end.isoformat()}."


def clear_time_range() -> str:
    """Remove the publication-date filter so retrieval covers all indexed papers.

    Call this when the user explicitly asks to ignore date constraints
    (for example: "search all papers", "no date filter", "ignore the time
    range").

    Returns:
        Confirmation string.
    """
    time_range_state.clear()
    return "Time range cleared. Searching all indexed papers."


def download_cited_papers(
    source_paper_id: str, citation_indices: list[int]
) -> str:
    """Download papers cited by a specific source paper.

    Use this ONLY when the user explicitly asks to fetch the references
    cited by a particular paper that is NOT already shown in the retrieved
    context (for example: "download all papers cited by arXiv:2103.12345",
    or "fetch the references of the ViT paper" when ViT is not in our
    current retrieval). The application automatically downloads citations
    detected in the retrieved context via a deterministic regex path —
    do not call this tool for those cases.

    Args:
        source_paper_id: arXiv id of the source paper containing the
            citations (e.g. "2103.12345").
        citation_indices: 1-indexed citation numbers as they appear in the
            source paper's References section (e.g. [1, 5, 12]).

    Returns:
        Status string. Actual download and ingestion happen asynchronously;
        the user can monitor progress in the sidebar download log.
    """
    if not source_paper_id or not citation_indices:
        return "Error: source_paper_id and citation_indices are both required."

    try:
        indices = [int(i) for i in citation_indices]
    except (TypeError, ValueError):
        return "Error: citation_indices must be a list of integers."

    outcome = enqueue_citation_download(source_paper_id, indices)
    queued = outcome.get("queued", 0)
    skipped = outcome.get("skipped", 0)
    if queued == 0 and skipped > 0:
        return (
            f"No new download queued for {source_paper_id}: "
            f"{outcome.get('reason', 'duplicate request')}."
        )
    return (
        f"Queued background download of {queued} reference(s) cited by "
        f"{source_paper_id} (indices: {indices}). Check the sidebar "
        "download log for results."
    )


def get_tools() -> list:
    """Return the list of tool callables to register with Gemini AFC."""
    return [set_time_range, clear_time_range, download_cited_papers]
