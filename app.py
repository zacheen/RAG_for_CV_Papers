"""Streamlit RAG chatbot for arXiv computer vision papers."""

import streamlit as st

from src.config import OLLAMA_MODEL, TOP_K
from src.processing.embedder import get_collection_stats
from src.rag.generator import generate_answer_stream
from src.rag.retriever import format_context, retrieve

WEEKLY_SUMMARY_PROMPT = (
    "Please summarize the newest computer vision papers from the last 7 days. "
    "Group the answer into: 1) key themes, 2) notable papers, 3) why they matter, "
    "and 4) open questions. Only use the retrieved papers and explicitly mention "
    "when the context is insufficient."
)


def run_query(prompt: str, top_k: int, recent_days: int | None, model_name: str) -> None:
    """Execute one RAG query and append the result to session history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            results = retrieve(prompt, top_k=top_k, recent_days=recent_days)
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
            stream = generate_answer_stream(
                question=prompt,
                context=context,
                model=model_name,
                chat_history=history,
            )
            response = st.write_stream(stream)
        except Exception as e:
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

with st.sidebar:
    st.header("Settings")
    your_name = st.text_input("Your name")
    top_k = st.slider("Number of retrieved chunks", 1, 20, TOP_K)
    if st.button("Summarize new papers from last 7 days", use_container_width=True):
        st.session_state["recent_only"] = True
        st.session_state["recent_scope_locked"] = True
        st.session_state["pending_prompt"] = WEEKLY_SUMMARY_PROMPT
        st.session_state["pending_top_k"] = max(top_k, 12)
    if st.session_state["recent_scope_locked"]:
        st.caption("Recent-only mode is locked for this summary follow-up chat.")
        if st.button("Unlock recent-only mode", use_container_width=True):
            st.session_state["recent_scope_locked"] = False
    recent_only = st.checkbox(
        "Only search recent 7 days CV recommendations",
        key="recent_only",
        disabled=st.session_state["recent_scope_locked"],
    )
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
if active_prompt:
    recent_days = 7 if recent_only else None
    run_query(active_prompt, top_k=active_top_k, recent_days=recent_days, model_name=model_name)
