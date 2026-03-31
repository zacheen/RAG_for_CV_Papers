"""Streamlit RAG chatbot for arXiv computer vision papers."""

import streamlit as st

from src.config import OLLAMA_MODEL, TOP_K
from src.processing.embedder import get_collection, get_collection_stats
from src.rag.retriever import retrieve, format_context
from src.rag.generator import generate_answer_stream

# --- Page config ---
st.set_page_config(page_title="CV Paper RAG", page_icon="📄", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    your_name = st.text_input("Your name")
    top_k = st.slider("Number of retrieved chunks", 1, 20, TOP_K)
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

# --- Main area ---
if your_name:
    st.title(f"Hi {your_name} — Ask about CV papers")
else:
    st.title("Computer Vision Paper RAG 📄")

st.caption(f"Powered by {model_name} via Ollama + ChromaDB retrieval")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ask me anything about computer vision research papers!"}
    ]

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    title = src.get("title", "Unknown")
                    url = src.get("arxiv_url", "")
                    authors = src.get("authors", "")
                    passage = src.get("passage", "")
                    distance = src.get("distance", 0)
                    if url:
                        st.markdown(f"**[{title}]({url})** (similarity: {1 - distance:.2%})")
                    else:
                        st.markdown(f"**{title}** (similarity: {1 - distance:.2%})")
                    if authors:
                        st.caption(authors)
                    if passage:
                        st.markdown(f"> {passage[:200]}{'...' if len(passage) > 200 else ''}")

# --- Chat input ---
if prompt := st.chat_input("Ask a question about computer vision papers..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve relevant chunks
    with st.chat_message("assistant"):
        try:
            results = retrieve(prompt, top_k=top_k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}. Make sure you've run the ingestion script.")
            st.stop()

        if not results:
            response = "I couldn't find any relevant papers. Make sure the corpus has been ingested."
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            context = format_context(results)

            # Build chat history for multi-turn (last 6 messages max)
            history = []
            for m in st.session_state.messages[-6:]:
                if m["role"] in ("user", "assistant") and "sources" not in m:
                    history.append({"role": m["role"], "content": m["content"]})

            # Stream the response
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

            # Show sources with clickable citations
            sources = [
                {
                    "title": r["title"],
                    "arxiv_url": r.get("arxiv_url", ""),
                    "authors": r.get("authors", ""),
                    "passage": r["text"][:300],
                    "distance": r["distance"],
                }
                for r in results
            ]
            with st.expander("Sources"):
                for src in sources:
                    url = src.get("arxiv_url", "")
                    if url:
                        st.markdown(f"**[{src['title']}]({url})** (similarity: {1 - src['distance']:.2%})")
                    else:
                        st.markdown(f"**{src['title']}** (similarity: {1 - src['distance']:.2%})")
                    if src.get("authors"):
                        st.caption(src["authors"])
                    st.markdown(f"> {src['passage']}{'...' if len(src['passage']) >= 300 else ''}")
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
            })
