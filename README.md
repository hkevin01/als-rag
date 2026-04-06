# ALS-RAG: ALS Research Literature RAG System

A Retrieval-Augmented Generation (RAG) system specialized for **Amyotrophic Lateral Sclerosis (ALS)** research. Ingests scientific literature from PubMed, Semantic Scholar, and arXiv, indexes it using FAISS vector search, and provides AI-powered evidence-based answers via a Streamlit web interface or CLI.

## Architecture

```
Literature Sources          Ingestion Pipeline         Storage
PubMed ──────────┐          ┌─ NER Extraction           FAISS IndexFlatIP
Semantic Scholar ─┤─→ Chunk ─┤─ Embedding               JSON Metadata
arXiv ───────────┘          └─ FAISS Index
                                    │
Query ──→ HybridRetriever ──→ Dense Search + Query Expansion ──→ GPT-4o-mini ──→ Answer
```

## Features

- **ALS-specific ingestion**: 18+ PubMed queries, 10 Semantic Scholar queries, 7 arXiv queries targeting ALS research
- **Domain NER**: Extracts genes (SOD1, C9orf72, FUS, TARDBP...), biomarkers (NfL, TDP-43...), clinical scales (ALSFRS-R, FVC...), drugs (riluzole, tofersen...), subtypes
- **Clinical feature matching**: Input ALSFRS-R, FVC, onset phenotype → matched literature queries
- **Hybrid retrieval**: Dense sentence-transformer embeddings + ALS synonym query expansion
- **Evidence-grounded generation**: GPT-4o-mini with ALS expert system prompt and citation requirements
- **Streamlit UI**: Multi-page — Search, Corpus Stats, Clinical Features, About
- **CLI**: `als-rag "What is the role of NfL as an ALS biomarker?"`

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hkevin01/als-rag.git
cd als-rag

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Install (requires uv)
make install

# 4. Ingest literature corpus
make ingest

# 5. Launch web UI
make run

# Or use CLI
als-rag "What is the efficacy of tofersen in SOD1-ALS?"
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for generation) | OpenAI API key |
| `PUBMED_API_KEY` | No (rate limit) | NCBI E-utilities API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No (rate limit) | Semantic Scholar key |
| `CONTACT_EMAIL` | Yes (PubMed ToS) | Your email for PubMed API |

## Project Structure

```
als-rag/
├── src/als_rag/
│   ├── ingestion/          # PubMed, Scholar, arXiv clients + pipeline
│   ├── retrieval/          # Dense, hybrid retriever, query expander
│   ├── storage/            # FAISS vector DB wrapper
│   ├── generation/         # OpenAI RAG generator
│   ├── signals/            # ALS clinical feature extractor
│   ├── web_ui/             # Streamlit app + pages
│   └── cli/                # Command-line interface
├── tests/                  # pytest test suite
├── data/embeddings/        # FAISS index (generated)
├── Makefile
└── pyproject.toml
```

## ALS Domain Coverage

| Category | Examples |
|---|---|
| Genes | SOD1, C9orf72, FUS, TARDBP, TBK1, NEK1, UBQLN2 |
| Biomarkers | Neurofilament light chain (NfL), TDP-43, phospho-NfH, YKL-40 |
| Clinical scales | ALSFRS-R, FVC, King's staging, MiToS staging, El Escorial |
| Treatments | Riluzole, Edaravone, Tofersen (BIIB067), AMX0035 |
| Phenotypes | Bulbar onset, limb onset, respiratory onset, ALS-FTD, PLS, PMA |

## Development

```bash
make test        # Run pytest
make lint        # Ruff linting
make format      # Black formatting
make typecheck   # mypy type checking
make clean       # Remove FAISS index and caches
```

## Based On

Architecture adapted from [eeg-rag](https://github.com/hkevin01/eeg-rag), adapted for the ALS research domain.
