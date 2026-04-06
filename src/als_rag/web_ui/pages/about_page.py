"""About page."""

import streamlit as st


def render():
    st.title("About ALS-RAG")
    st.markdown("""
## ALS Research Literature RAG System

**ALS-RAG** is a Retrieval-Augmented Generation system for ALS (Amyotrophic Lateral Sclerosis) research.
It ingests scientific literature from PubMed, Semantic Scholar, and arXiv, indexes it with FAISS,
and provides AI-powered answers grounded in the literature.

### Features
- **Ingestion**: PubMed, Semantic Scholar, arXiv sources with ALS-specific queries
- **Retrieval**: Hybrid dense (sentence-transformers) + query expansion
- **Generation**: GPT-4o-mini with evidence-based citations
- **Clinical features**: ALSFRS-R, FVC, onset phenotype → matched literature

### Key ALS Domains
- Genetics: SOD1, C9orf72, FUS, TARDBP, TBK1, NEK1, UBQLN2
- Biomarkers: Neurofilament light chain, TDP-43, phospho-NfH
- Clinical scales: ALSFRS-R, FVC, King's staging, MiToS staging
- Treatments: Riluzole, Edaravone, Tofersen, AMX0035

### Quick Start
```bash
make install
make ingest
make run
```
""")
