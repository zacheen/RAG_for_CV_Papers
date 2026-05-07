"""Single-paper ingest pipeline reusable from background threads.

Mirrors the inner loop of :mod:`scripts.ingest` but accepts one paper at a
time (by arXiv id) and returns a status dict instead of printing.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from src.config import PDF_DIR
from src.ingestion.arxiv_downloader import ARXIV_API_URL, NAMESPACE, download_pdf
from src.ingestion.pdf_parser import parse_pdf
from src.processing.chunker import chunk_document
from src.processing.embedder import index_chunks


def _fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    """Look up arXiv metadata for a single paper id via the arXiv API."""
    params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    url = f"{ARXIV_API_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    entry = root.find("atom:entry", NAMESPACE)
    if entry is None:
        return None

    id_el = entry.find("atom:id", NAMESPACE)
    title_el = entry.find("atom:title", NAMESPACE)
    summary_el = entry.find("atom:summary", NAMESPACE)
    published_el = entry.find("atom:published", NAMESPACE)
    if id_el is None or title_el is None:
        return None

    paper_id = id_el.text.strip().split("/abs/")[-1]
    authors = [
        a.find("atom:name", NAMESPACE).text.strip()
        for a in entry.findall("atom:author", NAMESPACE)
        if a.find("atom:name", NAMESPACE) is not None
    ]
    pdf_link = None
    for link in entry.findall("atom:link", NAMESPACE):
        if link.attrib.get("title") == "pdf":
            pdf_link = link.attrib["href"]
            break
    if pdf_link is None:
        pdf_link = f"http://arxiv.org/pdf/{paper_id}"

    return {
        "id": paper_id,
        "title": (title_el.text or "").strip().replace("\n", " "),
        "summary": (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else "",
        "authors": authors,
        "published": (published_el.text or "").strip() if published_el is not None else "",
        "pdf_url": pdf_link,
    }


def ingest_paper(arxiv_id: str, *, output_dir: Path = PDF_DIR) -> dict:
    """Download (if needed), parse, chunk, and index a single arXiv paper.

    Args:
        arxiv_id: arXiv id (with or without slashes).
        output_dir: Directory used for the PDF cache.

    Returns:
        Dict with keys:
            - ``status``: ``"ok"`` | ``"failed"``
            - ``arxiv_id``: echoed input
            - ``chunks_indexed``: int (0 on failure)
            - ``reason``: human-readable explanation on failure
    """
    meta = _fetch_arxiv_metadata(arxiv_id)
    if meta is None:
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "chunks_indexed": 0,
            "reason": "arXiv metadata lookup failed",
        }

    pdf_path = download_pdf(meta, output_dir=output_dir)
    if pdf_path is None:
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "chunks_indexed": 0,
            "reason": "pdf download failed",
        }

    try:
        parsed = parse_pdf(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "chunks_indexed": 0,
            "reason": f"pdf parse failed: {exc}",
        }

    text = parsed.get("text", "")
    if len(text.strip()) < 100:
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "chunks_indexed": 0,
            "reason": "too little text extracted",
        }

    arxiv_url = meta.get("pdf_url", f"https://arxiv.org/abs/{meta['id']}").replace(
        "/pdf/", "/abs/"
    )
    chunks = chunk_document(
        text,
        paper_id=meta["id"],
        title=meta.get("title", meta["id"]),
        arxiv_url=arxiv_url,
        authors=", ".join(meta.get("authors", [])),
        published=meta.get("published", ""),
        abstract=meta.get("summary", ""),
    )

    try:
        indexed = index_chunks(chunks)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "chunks_indexed": 0,
            "reason": f"index failed: {exc}",
        }

    return {
        "status": "ok",
        "arxiv_id": arxiv_id,
        "chunks_indexed": indexed,
        "reason": "",
    }
