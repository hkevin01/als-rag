"""ALS-RAG Streamlit web application entry point."""

import streamlit as st

st.set_page_config(
    page_title="ALS-RAG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("🧠 ALS-RAG")
st.sidebar.caption("ALS Research Literature Assistant")
page = st.sidebar.radio(
    "Navigate",
    ["Search", "Corpus Stats", "Clinical Features", "About"],
)

# ── Page routing ───────────────────────────────────────────────────────────
if page == "Search":
    from als_rag.web_ui.pages.search_page import render
    render()
elif page == "Corpus Stats":
    from als_rag.web_ui.pages.corpus_page import render
    render()
elif page == "Clinical Features":
    from als_rag.web_ui.pages.clinical_page import render
    render()
else:
    from als_rag.web_ui.pages.about_page import render
    render()
