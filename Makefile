.PHONY: install ingest run test lint format clean review verify

PYTHON := python
UV := uv

install:
	$(UV) pip install -e ".[dev,web]"

.env:
	cp .env.example .env
	@echo "Edit .env with your API keys"

ingest: .env
	$(PYTHON) -m als_rag.cli.query --ingest

ingest-all: .env
	$(PYTHON) -m als_rag.cli.query --ingest --sources pubmed,scholar,arxiv,clinicaltrials,europepmc

run: .env
	streamlit run src/als_rag/web_ui/app.py

query: .env
	$(PYTHON) -m als_rag.cli.query "$(Q)"

review: .env
	$(PYTHON) -m als_rag.cli.query --review "$(T)"

verify: .env
	$(PYTHON) -m als_rag.cli.query "$(Q)" --verify

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/als_rag/

clean:
	rm -rf data/embeddings/als_faiss.index data/embeddings/als_metadata.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "ALS-RAG Makefile targets:"
	@echo "  install      - Install with uv"
	@echo "  ingest       - Fetch and index ALS literature (all 5 sources)"
	@echo "  ingest-all   - Same as ingest (explicit all-sources)"
	@echo "  run          - Launch Streamlit UI"
	@echo "  query Q=     - CLI research query"
	@echo "  review T=    - Systematic mini-review on topic T"
	@echo "  verify Q=    - Query then run citation verification"
	@echo "  test         - Run tests"
	@echo "  clean        - Remove FAISS index"
