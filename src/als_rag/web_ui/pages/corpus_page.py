"""Corpus statistics page."""

import streamlit as st
import json


def render():
    st.title("📊 Corpus Statistics")
    from als_rag.utils.config import get_config
    config = get_config()

    if not config.faiss_index_path.exists():
        st.info("No corpus indexed yet. Run `make ingest` to build the index.")
        return

    import faiss
    index = faiss.read_index(str(config.faiss_index_path))
    total_vectors = index.ntotal

    with open(config.faiss_metadata_path) as f:
        metadata = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total chunks", total_vectors)
    col2.metric("Unique articles", len({m.get("doc_id", "") for m in metadata}))
    col3.metric("Embedding model", config.embedding_model)

    sources = {}
    years = {}
    for m in metadata:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        yr = m.get("year", "unknown")
        years[yr] = years.get(yr, 0) + 1

    st.subheader("Sources")
    st.bar_chart(sources)

    st.subheader("Publication years")
    sorted_years = dict(sorted(years.items()))
    st.bar_chart(sorted_years)
