.PHONY: install ingest run test lint format clean

PYTHON := python
UV := uv

install:
	$(UV) pip install -e ".[dev,web]"

.env:
	cp .env.example .env
	@echo "Edit .env with your API keys"

ingest: .env
	$(PYTHON) -m als_rag.cli.query --ingest

run: .env
	streamlit run src/als_rag/web_ui/app.py

query: .env
	$(PYTHON) -m als_rag.cli.query "$(Q)"

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
	@echo "  install   - Install with uv"
	@echo "  ingest    - Fetch and index ALS literature"
	@echo "  run       - Launch Streamlit UI"
	@echo "  query Q=  - CLI query"
	@echo "  test      - Run tests"
	@echo "  clean     - Remove FAISS index"
