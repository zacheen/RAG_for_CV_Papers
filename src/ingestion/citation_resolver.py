"""Resolve a paper's reference list via Semantic Scholar.

Used by the cited-paper download feature to translate inline ``[N]`` citation
indices into actual arXiv ids that the existing ingestion pipeline can
consume.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from src.ingestion.arxiv_downloader import ARXIV_API_URL, NAMESPACE

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper"
REQUEST_TIMEOUT_SECONDS = 30
_VERSION_SUFFIX = re.compile(r"v\d+$")
_NORMALISE_RE = re.compile(r"[^a-z0-9 ]+")


class CitationResolverError(RuntimeError):
    """Raised when the Semantic Scholar lookup fails."""


def _strip_version(arxiv_id: str) -> str:
    """Remove a trailing version suffix (e.g. ``v2``) so Semantic Scholar
    accepts the lookup. The API rejects ``arXiv:2507.05963v2`` with HTTP 404
    but happily resolves ``arXiv:2507.05963``.
    """
    return _VERSION_SUFFIX.sub("", arxiv_id)


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
    paper_ref = f"arXiv:{_strip_version(arxiv_id)}"
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


def _normalise_title(title: str) -> str:
    """Lowercase + strip non-alphanumerics for fuzzy title matching."""
    return _NORMALISE_RE.sub(" ", title.lower()).strip()


def search_arxiv_by_title(title: str, *, max_candidates: int = 5) -> str | None:
    """Look up an arXiv paper id by title.

    Used as a fallback when Semantic Scholar returns a reference without
    its ``externalIds.ArXiv`` field (common for papers SS hasn't bothered to
    cross-link). The arXiv API supports Lucene-style ``ti:`` queries; we
    take the top ``max_candidates`` and return the first whose normalised
    title is a near-match of the input.

    Args:
        title: The reference title (free text from Semantic Scholar).
        max_candidates: Cap on results to inspect.

    Returns:
        arXiv id string (with slash form preserved, e.g. ``"1312.6114"``)
        or ``None`` if no confident match was found.
    """
    clean = (title or "").strip().rstrip(".").strip()
    if len(clean) < 10:
        return None

    params = urllib.parse.urlencode(
        {
            "search_query": f'ti:"{clean}"',
            "max_results": max_candidates,
        }
    )
    url = f"{ARXIV_API_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    target = _normalise_title(clean)
    if not target:
        return None

    for entry in root.findall("atom:entry", NAMESPACE):
        title_el = entry.find("atom:title", NAMESPACE)
        id_el = entry.find("atom:id", NAMESPACE)
        if title_el is None or id_el is None:
            continue
        result_title = (title_el.text or "").strip().replace("\n", " ")
        normalised = _normalise_title(result_title)
        if not normalised:
            continue
        if normalised == target or normalised.startswith(target) or target.startswith(normalised):
            return id_el.text.strip().split("/abs/")[-1]
    return None


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
