.PHONY: help install install-dev install-data run-ui run-mcp seed vector clean test lint

help:
	@echo "Available targets:"
	@echo "  make install       - base setup (venv + deps + dirs)"
	@echo "  make install-dev   - base setup + dev deps"
	@echo "  make install-data  - base setup + ingest data (DB + vectors)"
	@echo "  make seed          - populate SQLite from CSV"
	@echo "  make vector        - build Chroma vector store from PDFs"
	@echo "  make run-ui        - launch Streamlit UI"
	@echo "  make run-mcp       - start the MCP server"
	@echo "  make test          - run pytest"
	@echo "  make clean         - remove venv, caches, and generated data"

install:
	bash install.sh

install-dev:
	bash install.sh --dev

install-data:
	bash install.sh --with-data

seed:
	python -m src.database.seed_data

vector:
	python -c "from pathlib import Path; from src.ingestion.pdf_loader import load_and_chunk_pdf; from src.ingestion.vector_store import build_vector_store; c=[]; [c.extend(load_and_chunk_pdf(str(p))) for p in Path('data/raw/policy_documents').glob('*.pdf')]; build_vector_store(c)"

run-ui:
	streamlit run src/ui/app.py

run-mcp:
	python -m src.mcp_server.server

test:
	pytest -q

clean:
	rm -rf .venv .pytest_cache **/__pycache__ data/processed/*.db data/processed/chroma_db