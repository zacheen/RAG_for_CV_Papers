"""Resolve a paper's reference list via Semantic Scholar.

Used by the cited-paper download feature to translate inline ``[N]`` citation
indices into actual arXiv ids that the existing ingestion pipeline can
consume.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper"
REQUEST_TIMEOUT_SECONDS = 30


class CitationResolverError(RuntimeError):
    """Raised when the Semantic Scholar lookup fails."""


def resolve_references(arxiv_id: str, *, max_refs: int = 200) -> list[dict]:
    """Fetch the reference list for an arXiv paper.

    Args:
        arxiv_id: arXiv id with or without slashes (e.g. ``"2103.12345"`` or
            ``"cs/0102001"``).
        max_refs: Cap on the number of references to request.

    Returns:
        List of reference dicts in the order Semantic Scholar returns them
        (matching the order they appear in the paper's References section, so
        index ``i`` in the LLM's ``[i]`` corresponds to entry ``i - 1`` in
        this list). Each dict has at minimum:
            - ``arxiv_id``: str | None
            - ``title``: str
            - ``paper_id_ss``: Semantic Scholar paper id (may be empty)

    Raises:
        CitationResolverError: On network / parsing failures.
    """
    paper_ref = f"arXiv:{arxiv_id}"
    params = urllib.parse.urlencode(
        {
            "fields": "externalIds,title",
            "limit": max_refs,
        }
    )
    url = f"{SEMANTIC_SCHOLAR_BASE}/{urllib.parse.quote(paper_ref, safe=':')}/references?{params}"

    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise CitationResolverError(
            f"Semantic Scholar HTTP {exc.code} for {paper_ref}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CitationResolverError(
            f"Semantic Scholar network error for {paper_ref}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CitationResolverError(
            f"Semantic Scholar returned non-JSON for {paper_ref}"
        ) from exc

    data = payload.get("data") or []
    refs: list[dict] = []
    for entry in data:
        cited = (entry or {}).get("citedPaper") or {}
        external = cited.get("externalIds") or {}
        refs.append(
            {
                "arxiv_id": external.get("ArXiv"),
                "title": cited.get("title", ""),
                "paper_id_ss": cited.get("paperId", ""),
            }
        )
    return refs


def pick_references(
    references: list[dict], indices: list[int]
) -> list[tuple[int, dict | None]]:
    """Map 1-indexed citation numbers to entries in ``references``.

    Args:
        references: Output of :func:`resolve_references`.
        indices: 1-indexed citation numbers as they appear in the source paper.

    Returns:
        List of ``(index, entry_or_None)``. Entry is ``None`` when the index
        is out of range.
    """
    picked: list[tuple[int, dict | None]] = []
    for idx in indices:
        if 1 <= idx <= len(references):
            picked.append((idx, references[idx - 1]))
        else:
            picked.append((idx, None))
    return picked
