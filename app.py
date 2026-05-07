"""Streamlit RAG chatbot for arXiv computer vision papers."""

import datetime
import sys
import threading
import time

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.config import GEMINI_MODEL, LLM_BACKEND, OLLAMA_MODEL, TOP_K
from src.processing.embedder import (
    get_collection,
    get_collection_lite,
    get_collection_stats,
    is_embedder_loaded,
)
from src.rag import download_state
from src.rag.download_state import (
    enqueue_citation_download,
    extract_inline_citations,
    get_log_snapshot,
    is_busy,
)
from src.rag.generator import generate_answer_stream as ollama_generate_answer_stream
from src.rag import generator_gemini, tools as rag_tools
from src.rag.generator_gemini import (
    generate_answer_stream as gemini_generate_answer_stream,
    run_pre_rag_pass,
)
from src.rag.retriever import format_context, retrieve, retrieve_recent_papers
from src.rag.tools import retrieval_query_state, time_range_state


@st.cache_resource
def get_cached_lite_collection():
    """Lightweight collection (no SentenceTransformer) for metadata-only ops:
    sidebar chunk count, citation dedup checks. Cold cost ~0.4s, so cheap
    enough to live on the first-paint path."""
    return get_collection_lite()


@st.cache_resource(show_spinner=False)
def get_cached_collection():
    """Full collection with SentenceTransformer embedder. Used by retrieve()
    and recent-paper lookups. Cold cost ~17s for the first call (loads the
    embedder from disk), then cached for the rest of the Streamlit process
    lifetime.

    ``show_spinner`` is disabled because this function is normally called
    from the background warmup thread at module load — surfacing a spinner
    there would render at the top of the page (where the warmup kickoff
    lives in script flow) and overlap the title. Streamlit's built-in
    top-right "Running..." indicator still fires if the user submits a
    prompt while warmup is still in flight."""
    return get_collection()


# Inject the cached lite collection into download_state so its background
# citation-dedup checks reuse the same handle without download_state having
# to import Streamlit. Only the lite collection is needed there because
# _is_in_db only does metadata lookups.
download_state.set_lite_collection_provider(get_cached_lite_collection)


@st.cache_resource
def _kickoff_embedder_warmup() -> bool:
    """Pre-warm the heavy embedder cache in a daemon thread on first run.

    The user is going to retrieve() eventually, which needs the
    SentenceTransformer model (~17s cold load). Loading it in a background
    thread while the user reads the page / types their first prompt usually
    means by the time they submit, ``get_cached_collection()`` is already
    cache-warm and the answer streams immediately.

    Streamlit's ``@st.cache_resource`` is thread-safe: if the user submits
    before the warmup finishes, their main-thread ``get_cached_collection()``
    call blocks on the same internal lock and reuses the in-progress result
    — no double computation.

    Wrapped in ``@st.cache_resource`` itself so the kicker runs exactly once
    per Streamlit process (script reruns hit the cached return and skip the
    spawn). Returns a sentinel so Streamlit can cache it.
    """
    def _warmup() -> None:
        try:
            get_cached_collection()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[embedder-warmup] failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=_warmup, daemon=True, name="embedder-warmup")
    # Attach the current Streamlit ScriptRunContext to the worker so it can
    # safely interact with @st.cache_resource without the "missing
    # ScriptRunContext" warning that Streamlit otherwise logs to stderr.
    add_script_run_ctx(thread)
    thread.start()
    return True


_kickoff_embedder_warmup()


def get_generate_answer_stream(backend: str):
    """Return the streaming generator function for the chosen backend."""
    if backend == "gemini":
        return gemini_generate_answer_stream
    return ollama_generate_answer_stream


WEEKLY_SUMMARY_PROMPT = (
    "Please summarize the newest computer vision papers from the last 7 days. "
    "Group the answer into: 1) key themes, 2) notable papers, 3) why they matter, "
    "and 4) open questions. Only use the retrieved papers and explicitly mention "
    "when the context is insufficient."
)


def _sync_time_range_to_module() -> None:
    """Push session_state time range into the module-level state used by tools."""
    tr = st.session_state.get("time_range") or {}
    time_range_state.start_date = tr.get("start_date")
    time_range_state.end_date = tr.get("end_date")


def _sync_time_range_from_module() -> None:
    """Pull module-level state back into session_state after the pre-RAG pass."""
    st.session_state["time_range"] = time_range_state.to_dict()


def _wait_for_downloads(max_seconds: float = 60.0) -> None:
    """Block until all background ingestion jobs finish, with a spinner."""
    if not is_busy():
        return
    deadline = time.monotonic() + max_seconds
    with st.spinner("Waiting for background paper downloads to finish..."):
        while is_busy() and time.monotonic() < deadline:
            time.sleep(0.3)


def _parse_iso(value: str | None) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _format_time_range_display() -> str:
    tr = st.session_state.get("time_range") or {}
    start, end = tr.get("start_date"), tr.get("end_date")
    if not start and not end:
        return "All time"
    if start and end:
        return f"{start} -> {end}"
    if start:
        return f"{start} -> ..."
    return f"... -> {end}"


def _download_attempt_footer(triggered_paths: list[str]) -> str:
    if not triggered_paths:
        return ""
    sources = " and ".join(triggered_paths)
    return (
        "\n\n---\n"
        f"_Attempted background download of cited papers ({sources}). "
        "If the answer above feels incomplete, the new papers may now be in "
        "the database — please ask again._"
    )


def run_query(
    prompt: str,
    top_k: int,
    model_name: str,
    retrieval_query: str | None = None,
    backend: str = "ollama",
) -> None:
    """Execute one RAG query and append the result to session history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Stage 0: block if a previous download/ingest is still running.
    _wait_for_downloads()

    triggered_paths: list[str] = []

    with st.chat_message("assistant"):
        with st.status("Processing your question...", expanded=True) as status:
            # Stage 1: pre-RAG function-calling pass (Gemini only).
            if backend == "gemini":
                status.update(label="Detecting query intent (date filters, citation requests)...")
                _sync_time_range_to_module()
                log_size_before = len(get_log_snapshot())
                in_flight_before = download_state._busy_count  # noqa: SLF001
                run_pre_rag_pass(prompt, model=model_name)
                _sync_time_range_from_module()
                if download_state._busy_count > in_flight_before or len(get_log_snapshot()) > log_size_before:  # noqa: SLF001
                    triggered_paths.append("path B / explicit request")
                    status.write("Tool call fired: explicit citation download requested")

            tr = st.session_state.get("time_range") or {}
            start_date = _parse_iso(tr.get("start_date"))
            end_date = _parse_iso(tr.get("end_date"))

            # Prefer the LLM's cleaned query (Stage 1 may have stripped time /
            # download noise). Fall back to caller-supplied retrieval_query,
            # then raw prompt.
            effective_retrieval_query = (
                retrieval_query
                or retrieval_query_state.cleaned
                or prompt
            )

            # Stage 2: retrieve. The label depends on whether the embedder is
            # already warm — first query in a process pays the SentenceTransformer
            # load (~17s) unless the background warmup beat them to it.
            if is_embedder_loaded():
                status.update(label="Retrieving relevant papers...")
            else:
                status.update(
                    label="Loading embedding model (first-time only, ~17s)..."
                )
            try:
                results = retrieve(
                    effective_retrieval_query,
                    top_k=top_k,
                    start_date=start_date,
                    end_date=end_date,
                    collection=get_cached_collection(),
                )
            except Exception as e:
                status.update(label=f"Retrieval failed: {e}", state="error", expanded=True)
                st.error(f"Retrieval failed: {e}. Make sure you've run the ingestion script.")
                st.stop()

            if not results:
                status.update(label="No relevant papers found", state="error", expanded=True)
                response = "I couldn't find any relevant papers. Make sure the corpus has been ingested."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            status.write(f"Retrieved {len(results)} papers")

            # Stage 2.5: regex auto-trigger (path A) — fire and forget.
            citation_map = extract_inline_citations(results)
            for source_paper_id, indices in citation_map.items():
                outcome = enqueue_citation_download(source_paper_id, indices)
                if outcome.get("queued", 0) > 0:
                    if "path A / inline citations" not in triggered_paths:
                        triggered_paths.append("path A / inline citations")
                    status.write(
                        f"Queued background citation download from "
                        f"`{source_paper_id}` for indices {indices}"
                    )

            context = format_context(results)

            history = []
            for message in st.session_state.messages[-6:]:
                if message["role"] in ("user", "assistant") and "sources" not in message:
                    history.append({"role": message["role"], "content": message["content"]})

            status.update(
                label=f"Retrieved {len(results)} papers — generating answer below",
                state="complete",
                expanded=False,
            )

        # Stage 3: stream final answer (no tools). Lives outside the status
        # block so the answer renders as the visible chat message body.
        try:
            generate_answer_stream = get_generate_answer_stream(backend)
            stream = generate_answer_stream(
                question=prompt,
                context=context,
                model=model_name,
                chat_history=history,
            )
            response = st.write_stream(stream)
        except Exception as e:
            if backend == "gemini":
                response = (
                    f"Generation failed: {e}. Is GOOGLE_API_KEY set and is "
                    f"{model_name} a valid Gemini model?"
                )
            else:
                response = f"Generation failed: {e}. Is Ollama running with {model_name}?"
            st.error(response)

        # Stage 4: append download footer when any path triggered.
        footer = _download_attempt_footer(triggered_paths)
        if footer:
            st.markdown(footer)
            response = (response or "") + footer

        sources = [
            {
                "title": result["title"],
                "arxiv_url": result.get("arxiv_url", ""),
                "authors": result.get("authors", ""),
                "passage": result["text"][:300],
                "distance": result["distance"],
            }
            for result in results
        ]
        with st.expander("Sources"):
            for source in sources:
                url = source.get("arxiv_url", "")
                if url:
                    st.markdown(
                        f"**[{source['title']}]({url})** "
                        f"(similarity: {1 - source['distance']:.2%})"
                    )
                else:
                    st.markdown(
                        f"**{source['title']}** "
                        f"(similarity: {1 - source['distance']:.2%})"
                    )
                if source.get("authors"):
                    st.caption(source["authors"])
                st.markdown(
                    f"> {source['passage']}"
                    f"{'...' if len(source['passage']) >= 300 else ''}"
                )
                st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
            }
        )

    # Force a rerun so the sidebar (Current search range, Pre-RAG debug,
    # Download log) reflects state changed by this turn — Streamlit only
    # re-renders the sidebar on a fresh script execution and the sidebar
    # block runs BEFORE run_query in the script flow.
    st.rerun()


def run_recent_summary(prompt: str, top_k: int, recent_days: int, model_name: str,
                       backend: str = "ollama") -> None:
    """Summarize recent papers directly from DB metadata instead of vector search."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    _wait_for_downloads()

    with st.chat_message("assistant"):
        with st.status(
            f"Looking up papers from the last {recent_days} days...",
            expanded=True,
        ) as status:
            # retrieve_recent_papers does a metadata-only date filter
            # (collection.get(where=...)) so it can use the lite collection
            # and skip the SentenceTransformer load entirely.
            try:
                results = retrieve_recent_papers(
                    recent_days=recent_days,
                    max_papers=top_k,
                    collection=get_cached_lite_collection(),
                )
            except Exception as e:
                status.update(label=f"Lookup failed: {e}", state="error", expanded=True)
                st.error(f"Recent paper lookup failed: {e}. Make sure you've run the ingestion script.")
                st.stop()

            if not results:
                status.update(
                    label=f"No indexed papers in the last {recent_days} days",
                    state="error",
                    expanded=True,
                )
                response = (
                    f"I couldn't find any indexed papers published in the last {recent_days} days. "
                    "Re-run ingestion to fetch newer arXiv papers."
                )
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            status.write(f"Found {len(results)} papers")

            context = format_context(results)

            history = []
            for message in st.session_state.messages[-6:]:
                if message["role"] in ("user", "assistant") and "sources" not in message:
                    history.append({"role": message["role"], "content": message["content"]})

            status.update(
                label=f"Found {len(results)} papers — generating summary below",
                state="complete",
                expanded=False,
            )

        try:
            generate_answer_stream = get_generate_answer_stream(backend)
            stream = generate_answer_stream(
                question=prompt,
                context=context,
                model=model_name,
                chat_history=history,
            )
            response = st.write_stream(stream)
        except Exception as e:
            if backend == "gemini":
                response = (
                    f"Generation failed: {e}. Is GOOGLE_API_KEY set and is "
                    f"{model_name} a valid Gemini model?"
                )
            else:
                response = f"Generation failed: {e}. Is Ollama running with {model_name}?"
            st.error(response)

        sources = [
            {
                "title": result["title"],
                "arxiv_url": result.get("arxiv_url", ""),
                "authors": result.get("authors", ""),
                "passage": result["text"][:300],
                "distance": result.get("distance", 0.0),
            }
            for result in results
        ]
        with st.expander("Sources"):
            for source in sources:
                url = source.get("arxiv_url", "")
                if url:
                    st.markdown(f"**[{source['title']}]({url})**")
                else:
                    st.markdown(f"**{source['title']}**")
                if source.get("authors"):
                    st.caption(source["authors"])
                st.markdown(
                    f"> {source['passage']}"
                    f"{'...' if len(source['passage']) >= 300 else ''}"
                )
                st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
            }
        )


st.set_page_config(page_title="CV Paper RAG", page_icon="CV", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Ask me anything about computer vision research papers!",
        }
    ]
if "time_range" not in st.session_state:
    st.session_state["time_range"] = {"start_date": None, "end_date": None}
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "pending_top_k" not in st.session_state:
    st.session_state["pending_top_k"] = None
if "pending_mode" not in st.session_state:
    st.session_state["pending_mode"] = "chat"

with st.sidebar:
    st.header("Settings")
    your_name = st.text_input("Your name")
    top_k = st.slider("Number of retrieved chunks", 1, 20, TOP_K)
    if st.button("Summarize new papers from last 7 days", use_container_width=True):
        st.session_state["pending_prompt"] = WEEKLY_SUMMARY_PROMPT
        st.session_state["pending_top_k"] = max(top_k, 12)
        st.session_state["pending_mode"] = "recent_summary"
        # Reflect the 7-day window in the sidebar's "Current search range"
        # without going through Gemini — this is a deterministic preset, not
        # a natural-language request that needs intent extraction.
        _today = datetime.date.today()
        st.session_state["time_range"] = {
            "start_date": (_today - datetime.timedelta(days=7)).isoformat(),
            "end_date": _today.isoformat(),
        }

    backend_options = ["ollama", "gemini"]
    default_backend_index = backend_options.index(LLM_BACKEND) if LLM_BACKEND in backend_options else 0
    backend = st.selectbox(
        "LLM backend",
        backend_options,
        index=default_backend_index,
        help="ollama = local/VM Ollama (original). gemini = Google AI Studio Gemini free tier.",
    )
    if backend == "gemini":
        model_name = st.text_input("Gemini model", value=GEMINI_MODEL)
        st.caption(
            "Set GOOGLE_API_KEY env var. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    else:
        model_name = st.text_input("Ollama model", value=OLLAMA_MODEL)

    st.divider()
    st.header("Current search range")
    st.write(_format_time_range_display())
    st.caption(
        "Set automatically when you mention dates in chat (Gemini backend). "
        "Say \"ignore the time filter\" to clear."
    )

    st.divider()
    st.header("Pre-RAG debug")
    if generator_gemini.last_pre_rag_error:
        st.error(generator_gemini.last_pre_rag_error)
    if retrieval_query_state.cleaned:
        st.caption("Cleaned retrieval query:")
        st.code(retrieval_query_state.cleaned, language="text")
    if rag_tools.last_call_log:
        st.caption("Last turn function calls:")
        for line in rag_tools.last_call_log:
            st.code(line, language="text")
    elif backend == "gemini" and not generator_gemini.last_pre_rag_error:
        st.caption("No tool calls fired in the last turn.")
    else:
        st.caption("(Gemini backend only)")

    st.divider()
    st.header("Download log")
    log_entries = list(reversed(get_log_snapshot()))
    if not log_entries:
        st.caption("No background downloads yet.")
    else:
        for entry in log_entries:
            icon = {
                "ok": "[OK]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "running": "[...]",
                "pending": "[...]",
            }.get(entry.status, "[?]")
            line = f"{icon} `{entry.arxiv_id}`"
            if entry.reason:
                line += f" — {entry.reason}"
            st.markdown(line)
    if is_busy():
        st.caption("Background ingestion in progress...")

    st.divider()
    with st.expander("Collection Stats", expanded=False):
        try:
            stats = get_collection_stats(collection=get_cached_lite_collection())
            st.metric("Indexed chunks", stats["total_chunks"])
        except Exception:
            st.warning("No indexed data yet. Run the ingestion script first.")

    st.divider()
    st.caption(
        "RAG system for arXiv CS.CV papers. "
        "Built for CS 6120 NLP Final Project."
    )

if your_name:
    st.title(f"Hi {your_name} - Ask about CV papers")
else:
    st.title("Computer Vision Paper RAG")

if backend == "gemini":
    st.caption(f"Powered by {model_name} via Google AI Studio + ChromaDB retrieval")
else:
    st.caption(f"Powered by {model_name} via Ollama + ChromaDB retrieval")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for source in msg["sources"]:
                    title = source.get("title", "Unknown")
                    url = source.get("arxiv_url", "")
                    authors = source.get("authors", "")
                    passage = source.get("passage", "")
                    distance = source.get("distance", 0)
                    if url:
                        st.markdown(f"**[{title}]({url})** (similarity: {1 - distance:.2%})")
                    else:
                        st.markdown(f"**{title}** (similarity: {1 - distance:.2%})")
                    if authors:
                        st.caption(authors)
                    if passage:
                        st.markdown(f"> {passage[:200]}{'...' if len(passage) > 200 else ''}")

prompt = st.chat_input("Ask a question about computer vision papers...")
active_prompt = st.session_state.pop("pending_prompt", None) or prompt
active_top_k = st.session_state.pop("pending_top_k", None) or top_k
active_mode = st.session_state.pop("pending_mode", "chat")
if active_prompt:
    if active_mode == "recent_summary":
        run_recent_summary(
            active_prompt,
            top_k=active_top_k,
            recent_days=7,
            model_name=model_name,
            backend=backend,
        )
    else:
        run_query(
            active_prompt,
            top_k=active_top_k,
            model_name=model_name,
            backend=backend,
        )
