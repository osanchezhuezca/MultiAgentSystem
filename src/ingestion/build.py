"""Build the Chroma vector store from policy PDFs.

Usage:
    python -m src.ingestion.build
    python -m src.ingestion.build --reset
    python -m src.ingestion.build --pdf-dir data/raw/policy_documents
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config import settings
from src.ingestion.pdf_loader import load_and_chunk_pdf
from src.ingestion.vector_store import build_vector_store


def build(pdf_dir: Path, persist_dir: Path, reset: bool = False) -> int:
    """Embed every PDF in pdf_dir into a Chroma store at persist_dir.

    Returns the number of chunks written.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    if reset and persist_dir.exists():
        print(f"Removing existing vector store at {persist_dir} ...")
        shutil.rmtree(persist_dir)

    all_chunks = []
    for pdf in pdfs:
        print(f"Loading {pdf.name} ...")
        chunks = load_and_chunk_pdf(str(pdf))
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced — check your PDFs.", file=sys.stderr)
        return 0

    print(f"Embedding {len(all_chunks)} chunks into {persist_dir} ...")
    build_vector_store(all_chunks, persist_dir=str(persist_dir))
    print(f"Done. Vector store contains {len(all_chunks)} chunks.")
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
        help="Where to persist the Chroma store",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing store before rebuilding",
    )
    args = parser.parse_args()

    build(args.pdf_dir, args.persist_dir, reset=args.reset)


if __name__ == "__main__":
    main()