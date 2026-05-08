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
_TITLE_MATCH_THRESHOLD = 0.7
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to",
    "with", "via", "from", "by", "is", "are", "as", "at",
}


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
            - ``pdf_url``: str | None — open-access PDF URL when SS has one

    Raises:
        CitationResolverError: On network / parsing failures.
    """
    paper_ref = f"arXiv:{_strip_version(arxiv_id)}"
    params = urllib.parse.urlencode(
        {
            "fields": "externalIds,title,openAccessPdf",
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
        oa = cited.get("openAccessPdf") or {}
        refs.append(
            {
                "arxiv_id": external.get("ArXiv"),
                "title": cited.get("title", ""),
                "paper_id_ss": cited.get("paperId", ""),
                "pdf_url": oa.get("url") or None,
            }
        )
    return refs


def _normalise_title(title: str) -> str:
    """Lowercase + strip non-alphanumerics for fuzzy title matching."""
    return _NORMALISE_RE.sub(" ", title.lower()).strip()


def _tokenise(text: str) -> set:
    """Tokenise a normalised title into significant tokens (no stopwords,
    no length-1 tokens) for set similarity comparison."""
    normalised = _normalise_title(text)
    return {
        tok
        for tok in normalised.split()
        if tok and tok not in _STOPWORDS and len(tok) > 1
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def search_arxiv_by_title(title: str, *, max_candidates: int = 8) -> dict:
    """Look up an arXiv paper id by title.

    Used as a fallback when Semantic Scholar returns a reference without
    its ``externalIds.ArXiv`` field (common for papers SS hasn't bothered to
    cross-link). The arXiv API supports Lucene-style ``ti:`` queries; we
    take the top ``max_candidates`` and pick the candidate with the highest
    Jaccard similarity over significant tokens (stopwords + 1-char tokens
    excluded). A match is confident when Jaccard >= ``_TITLE_MATCH_THRESHOLD``.

    Args:
        title: The reference title (free text from Semantic Scholar).
        max_candidates: Cap on results to inspect.

    Returns:
        Dict with keys:
        - ``arxiv_id``: str | None — the matching arXiv id, or ``None`` if
          no candidate cleared the similarity threshold.
        - ``best_candidate_title``: str — title of the closest arXiv result
          inspected (empty string if the API returned nothing).
        - ``best_score``: float — Jaccard similarity to the closest candidate.
        - ``candidates_inspected``: int — how many arXiv entries we looked at.
        - ``query``: str — the actual ti: query string we sent.
        - ``error``: str — populated when the lookup failed (network /
          parse), empty on success.
    """
    result = {
        "arxiv_id": None,
        "best_candidate_title": "",
        "best_score": 0.0,
        "candidates_inspected": 0,
        "query": "",
        "error": "",
    }

    clean = (title or "").strip().rstrip(".").strip()
    if len(clean) < 10:
        result["error"] = "title too short"
        return result

    query_str = f'ti:"{clean}"'
    result["query"] = query_str
    params = urllib.parse.urlencode(
        {"search_query": query_str, "max_results": max_candidates}
    )
    url = f"{ARXIV_API_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        result["error"] = f"arXiv API error: {exc}"
        return result

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        result["error"] = f"arXiv API parse error: {exc}"
        return result

    target_tokens = _tokenise(clean)
    if not target_tokens:
        result["error"] = "no significant tokens in title"
        return result

    best_score = 0.0
    best_id: "str | None" = None
    best_title = ""
    inspected = 0

    for entry in root.findall("atom:entry", NAMESPACE):
        title_el = entry.find("atom:title", NAMESPACE)
        id_el = entry.find("atom:id", NAMESPACE)
        if title_el is None or id_el is None:
            continue
        inspected += 1
        candidate_title = (title_el.text or "").strip().replace("\n", " ")
        candidate_tokens = _tokenise(candidate_title)
        score = _jaccard(target_tokens, candidate_tokens)
        if score > best_score:
            best_score = score
            best_title = candidate_title
            best_id = id_el.text.strip().split("/abs/")[-1]

    result["candidates_inspected"] = inspected
    result["best_candidate_title"] = best_title
    result["best_score"] = best_score
    if best_score >= _TITLE_MATCH_THRESHOLD:
        result["arxiv_id"] = best_id
    return result


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
