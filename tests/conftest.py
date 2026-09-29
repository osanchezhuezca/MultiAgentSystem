"""Shared pytest fixtures for the multi-agent test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "customer_support.db"
CHROMA_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "policy_documents"


# ---------- Availability checks ----------

def _db_exists() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def _chroma_exists() -> bool:
    return CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir())


def _pdfs_exist() -> bool:
    return PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf"))


# ---------- Skip markers ----------

requires_db = pytest.mark.skipif(
    not _db_exists(),
    reason="Seeded SQLite database not found. Run: python -m src.database.seed_data",
)

requires_chroma = pytest.mark.skipif(
    not _chroma_exists(),
    reason="Chroma vector store not found. Run: python -m src.ingestion.build",
)

requires_pdfs = pytest.mark.skipif(
    not _pdfs_exist(),
    reason="No policy PDFs found in data/raw/policy_documents/",
)


# ---------- Fixtures ----------

@pytest.fixture
def db_url() -> str:
    """SQLAlchemy URL for the seeded customer database."""
    return f"sqlite:///{DB_PATH}"


@pytest.fixture
def chroma_dir() -> str:
    """Absolute path to the Chroma persistence directory."""
    return str(CHROMA_PATH)


@pytest.fixture
def pdf_dir() -> Path:
    """Directory containing policy PDFs."""
    return PDF_DIR


@pytest.fixture
def empty_state() -> dict:
    """A fresh supervisor state with every field initialised.

    The supervisor graph crashes with KeyError if any field is missing,
    so tests must start from this canonical shape.
    """
    return {
        "query": "",
        "route": "",
        "sql_query": "",
        "rag_query": "",
        "sql_result": "",
        "rag_result": "",
        "final_answer": "",
    }
