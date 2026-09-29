"""Build the Chroma vector store from policy PDFs.

Usage:
    python -m src.ingestion.build
    python -m src.ingestion.build --reset
    python -m src.ingestion.build --pdf-dir data/raw/policy_documents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import settings
from src.ingestion.pdf_loader import load_and_chunk_pdf
from src.ingestion.vector_store import build_vector_store, reset_vector_store
from src.vectorstore import reset_client_cache


def build(pdf_dir: Path, persist_dir: Path | None = None, reset: bool = False) -> int:
    """Embed every PDF in pdf_dir into the shared Chroma collection.

    Returns the number of chunks written.

    The persist_dir argument is retained for CLI compatibility but is no
    longer used: the shared client in src/vectorstore.py always points at
    settings.chroma_persist_dir.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    if reset:
        print("Resetting vector store ...")
        reset_vector_store()
        reset_client_cache()

    all_chunks = []
    for pdf in pdfs:
        print(f"Loading {pdf.name} ...")
        chunks = load_and_chunk_pdf(str(pdf))
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced — check your PDFs.", file=sys.stderr)
        return 0

    print(f"Embedding {len(all_chunks)} chunks into the shared collection ...")
    build_vector_store(all_chunks)
    print(f"Done. Wrote {len(all_chunks)} chunks.")
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vector store.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(settings.policy_pdf_dir),
        help="Directory containing policy PDFs",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(settings.chroma_persist_dir),
        help="Retained for compatibility; the shared client owns the path.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before rebuilding",
    )
    args = parser.parse_args()

    build(args.pdf_dir, args.persist_dir, reset=args.reset)


if __name__ == "__main__":
    main()