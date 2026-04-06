"""Search & RAG answer page."""

import streamlit as st
from typing import List, Dict, Any


@st.cache_resource(show_spinner=False)
def _get_retriever():
    from als_rag.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever()


@st.cache_resource(show_spinner=False)
def _get_generator():
    from als_rag.generation.generator import ALSGenerator
    import os
    return ALSGenerator(api_key=os.environ.get("OPENAI_API_KEY", ""))


def render():
    st.title("🔍 ALS Research Search")
    st.caption("Ask a research question and get evidence-based answers from indexed ALS literature.")

    query = st.text_input(
        "Research question",
        placeholder="e.g. What is the role of C9orf72 repeat expansion in ALS?",
    )
    col1, col2 = st.columns([1, 3])
    top_k = col1.slider("Results", min_value=3, max_value=20, value=8)
    generate_answer = col2.checkbox("Generate AI answer (requires OpenAI key)", value=True)

    if st.button("Search", type="primary") and query:
        with st.spinner("Retrieving..."):
            retriever = _get_retriever()
            results: List[Dict[str, Any]] = retriever.retrieve(query, top_k=top_k)

        if not results:
            st.warning("No results found. Have you ingested papers? Run `make ingest`.")
            return

        if generate_answer:
            with st.spinner("Generating answer..."):
                generator = _get_generator()
                answer = generator.generate(query, results)
            st.subheader("Answer")
            st.markdown(answer)
            st.divider()

        st.subheader(f"Top {len(results)} relevant passages")
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            title = r.get("title", "Unknown")
            year = r.get("year", "")
            url = r.get("url", "")
            chunk = r.get("chunk_text", "")
            with st.expander(f"{i}. {title} ({year}) — score: {score:.3f}"):
                st.write(chunk)
                if url:
                    st.markdown(f"[View paper]({url})")
