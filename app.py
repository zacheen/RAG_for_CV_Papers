"""Streamlit RAG chatbot for arXiv computer vision papers."""

import streamlit as st

from src.config import GEMINI_MODEL, LLM_BACKEND, OLLAMA_MODEL, TOP_K
from src.processing.embedder import get_collection_stats
from src.rag.generator import generate_answer_stream as ollama_generate_answer_stream
from src.rag.generator_gemini import generate_answer_stream as gemini_generate_answer_stream
from src.rag.retriever import format_context, retrieve, retrieve_recent_papers


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
def run_query(
    prompt: str,
    top_k: int,
    recent_days: int | None,
    model_name: str,
    retrieval_query: str | None = None,
    backend: str = "ollama",
) -> None:
    """Execute one RAG query and append the result to session history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            results = retrieve(
                retrieval_query or prompt,
                top_k=top_k,
                recent_days=recent_days,
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}. Make sure you've run the ingestion script.")
            st.stop()

        if not results:
            response = "I couldn't find any relevant papers. Make sure the corpus has been ingested."
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return

        context = format_context(results)

        history = []
        for message in st.session_state.messages[-6:]:
            if message["role"] in ("user", "assistant") and "sources" not in message:
                history.append({"role": message["role"], "content": message["content"]})

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


def run_recent_summary(prompt: str, top_k: int, recent_days: int, model_name: str,
                       backend: str = "ollama") -> None:
    """Summarize recent papers directly from DB metadata instead of vector search."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            results = retrieve_recent_papers(recent_days=recent_days, max_papers=top_k)
        except Exception as e:
            st.error(f"Recent paper lookup failed: {e}. Make sure you've run the ingestion script.")
            st.stop()

        if not results:
            response = (
                f"I couldn't find any indexed papers published in the last {recent_days} days. "
                "Re-run ingestion to fetch newer arXiv papers."
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return

        context = format_context(results)

        history = []
        for message in st.session_state.messages[-6:]:
            if message["role"] in ("user", "assistant") and "sources" not in message:
                history.append({"role": message["role"], "content": message["content"]})

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
if "recent_only" not in st.session_state:
    st.session_state["recent_only"] = False
if "recent_scope_locked" not in st.session_state:
    st.session_state["recent_scope_locked"] = False
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
        st.session_state["recent_only"] = True
        st.session_state["recent_scope_locked"] = True
        st.session_state["pending_prompt"] = WEEKLY_SUMMARY_PROMPT
        st.session_state["pending_top_k"] = max(top_k, 12)
        st.session_state["pending_mode"] = "recent_summary"
    if st.session_state["recent_scope_locked"]:
        st.caption("Recent-only mode is locked for this summary follow-up chat.")
        if st.button("Unlock recent-only mode", use_container_width=True):
            st.session_state["recent_scope_locked"] = False
    recent_only = st.checkbox(
        "Only search recent 7 days CV recommendations",
        key="recent_only",
        disabled=st.session_state["recent_scope_locked"],
    )
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
    st.header("Collection Stats")
    try:
        stats = get_collection_stats()
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
    recent_days = 7 if recent_only else None
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
            recent_days=recent_days,
            model_name=model_name,
            backend=backend,
        )
