"""Clinical feature input → literature matched search."""

import streamlit as st


def render():
    st.title("🏥 Clinical Feature Search")
    st.caption("Input patient clinical features to find relevant ALS literature.")

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)
        alsfrs = col1.number_input("ALSFRS-R score (0-48)", min_value=0, max_value=48, value=40)
        fvc = col2.number_input("FVC % predicted", min_value=0, max_value=150, value=90)
        onset = st.selectbox("Onset phenotype", ["limb onset", "bulbar onset", "respiratory onset", "flail arm", "flail leg", "ALS-FTD"])
        gene = st.selectbox("Known gene variant", ["none", "SOD1", "C9orf72", "FUS", "TARDBP", "TBK1", "NEK1", "UBQLN2", "other"])
        nfl = st.number_input("Neurofilament light (pg/mL, 0=unknown)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Find relevant literature")

    if submitted:
        parts = [f"ALS {onset}"]
        if gene != "none":
            parts.append(f"{gene} variant ALS")
        if alsfrs < 30:
            parts.append("advanced ALS ALSFRS-R decline")
        if fvc < 70:
            parts.append("ALS respiratory insufficiency FVC")
        if nfl > 0:
            parts.append("neurofilament light chain ALS biomarker")
        query = " ".join(parts)

        st.info(f"Generated query: _{query}_")
        from als_rag.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        results = retriever.retrieve(query, top_k=8)
        if not results:
            st.warning("No relevant results found. Ingest papers first.")
        for i, r in enumerate(results, 1):
            with st.expander(f"{i}. {r.get('title', '')} ({r.get('year', '')})"):
                st.write(r.get("chunk_text", ""))
                url = r.get("url", "")
                if url:
                    st.markdown(f"[View paper]({url})")
