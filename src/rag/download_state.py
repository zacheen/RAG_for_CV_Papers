"""Background download/ingest job queue and dedup helpers.

Shared by both trigger paths for cited-paper download:

- Path A (regex auto): app.py extracts inline ``[X]`` citations from
  retrieved chunks via :func:`extract_inline_citations` and feeds each
  ``(source_paper_id, indices)`` pair to :func:`enqueue_citation_download`.
- Path B (LLM tool): the ``download_cited_papers`` tool calls the same
  :func:`enqueue_citation_download` directly when the user explicitly asks.

All state lives at module level so background threads can mutate it without
touching Streamlit's session state. The Streamlit UI reads
:func:`get_log_snapshot` / :func:`is_busy` on each rerun.
"""

from __future__ import annotations

import datetime
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import PDF_DIR
from src.processing.embedder import get_collection

LOG_MAX_ENTRIES = 10
INLINE_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class DownloadJobEntry:
    arxiv_id: str
    status: str  # "pending" | "running" | "ok" | "failed" | "skipped"
    reason: str = ""
    started_at: str = ""
    finished_at: str = ""
    source_paper_id: str = ""


_lock = threading.Lock()
_log: deque[DownloadJobEntry] = deque(maxlen=LOG_MAX_ENTRIES)
_in_flight: set[str] = set()  # arxiv_ids currently pending or running
_busy_count: int = 0  # number of background workers currently running


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _push_log(entry: DownloadJobEntry) -> None:
    with _lock:
        _log.append(entry)


def get_log_snapshot() -> list[DownloadJobEntry]:
    """Return a copy of the current download log (newest last)."""
    with _lock:
        return list(_log)


def is_busy() -> bool:
    """True iff at least one background download/ingest worker is running."""
    with _lock:
        return _busy_count > 0


def _is_in_db(arxiv_id: str) -> bool:
    """Check whether ChromaDB already has at least one chunk for arxiv_id."""
    try:
        collection = get_collection()
        # paper_id in chunk metadata uses the arXiv-id-with-slashes form.
        result = collection.get(where={"paper_id": arxiv_id}, limit=1)
        return bool(result and result.get("ids"))
    except Exception:
        return False


def _pdf_path_for(arxiv_id: str) -> Path:
    return PDF_DIR / f"{_safe_id(arxiv_id)}.pdf"


def _gate_check(arxiv_id: str) -> tuple[bool, str]:
    """Return (should_enqueue, reason).

    Runs the dedup gates in order:
      1. ChromaDB metadata
      2. PDF file already on disk
      3. Already in the in-flight set
    """
    if _is_in_db(arxiv_id):
        return False, "already in DB"
    if _pdf_path_for(arxiv_id).exists():
        # PDF exists but not in DB → still enqueue so the worker can re-ingest.
        return True, "re-index existing pdf"
    with _lock:
        if arxiv_id in _in_flight:
            return False, "in flight"
    return True, ""


def enqueue_citation_download(
    source_paper_id: str,
    citation_indices: list[int],
    *,
    worker: Callable[[str, list[int]], None] | None = None,
) -> dict:
    """Enqueue a background job that resolves citations and ingests papers.

    The worker is dispatched as a daemon thread and:
      1. Resolves ``citation_indices`` against the source paper's references
         via Semantic Scholar (handled inside the worker).
      2. For each resolved arXiv id, re-runs the dedup gates and downloads +
         ingests if still needed.
      3. Updates the module-level log + busy counter.

    Args:
        source_paper_id: arXiv id of the paper containing the citations.
        citation_indices: 1-indexed citation numbers as they appear in the
            source paper's text.
        worker: Optional override (used by tests). Defaults to the real worker
            in :mod:`src.rag.download_worker` (resolved lazily to avoid an
            import cycle).

    Returns:
        Dict summarizing the enqueue outcome.
    """
    citation_indices = sorted(set(int(i) for i in citation_indices))
    if not source_paper_id or not citation_indices:
        return {"queued": 0, "skipped": 0, "reason": "empty input"}

    if worker is None:
        from src.rag.download_worker import run_citation_download_job

        worker = run_citation_download_job

    job_key = f"{source_paper_id}::{','.join(str(i) for i in citation_indices)}"
    with _lock:
        if job_key in _in_flight:
            return {"queued": 0, "skipped": len(citation_indices), "reason": "job in flight"}
        _in_flight.add(job_key)
        global _busy_count
        _busy_count += 1

    def _run() -> None:
        try:
            worker(source_paper_id, citation_indices)
        finally:
            with _lock:
                _in_flight.discard(job_key)
                global _busy_count
                _busy_count -= 1

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "queued": len(citation_indices),
        "skipped": 0,
        "source_paper_id": source_paper_id,
    }


def mark_arxiv_in_flight(arxiv_id: str) -> bool:
    """Reserve an arxiv_id slot. Returns False if already in flight."""
    with _lock:
        if arxiv_id in _in_flight:
            return False
        _in_flight.add(arxiv_id)
        return True


def release_arxiv(arxiv_id: str) -> None:
    with _lock:
        _in_flight.discard(arxiv_id)


def record_result(
    arxiv_id: str,
    status: str,
    reason: str = "",
    source_paper_id: str = "",
    started_at: str = "",
) -> None:
    """Append a final result to the download log."""
    entry = DownloadJobEntry(
        arxiv_id=arxiv_id,
        status=status,
        reason=reason,
        started_at=started_at or _now_iso(),
        finished_at=_now_iso(),
        source_paper_id=source_paper_id,
    )
    _push_log(entry)


def gate_check(arxiv_id: str) -> tuple[bool, str]:
    """Public wrapper around the dedup gates for use by workers."""
    return _gate_check(arxiv_id)


def now_iso() -> str:
    return _now_iso()


def extract_inline_citations(retrieved_chunks: list[dict]) -> dict[str, list[int]]:
    """Scan retrieved chunks for ``[N]`` and ``[N, M]`` citation patterns.

    Args:
        retrieved_chunks: List of dicts with at least ``paper_id`` and ``text``
            keys (the same shape returned by :func:`src.rag.retriever.retrieve`).

    Returns:
        Mapping ``{source_paper_id: [sorted unique citation indices]}``.
        Chunks without a ``paper_id`` are skipped.
    """
    result: dict[str, set[int]] = {}
    for chunk in retrieved_chunks:
        source_id = chunk.get("paper_id", "")
        text = chunk.get("text", "")
        if not source_id or not text:
            continue
        for match in INLINE_CITATION_PATTERN.finditer(text):
            for piece in match.group(1).split(","):
                piece = piece.strip()
                if piece.isdigit():
                    result.setdefault(source_id, set()).add(int(piece))
    return {k: sorted(v) for k, v in result.items()}
