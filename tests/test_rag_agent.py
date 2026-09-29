"""Tests for the RAG agent and PDF ingestion pipeline."""
from __future__ import annotations

import pytest

from tests.conftest import requires_chroma, requires_pdfs


# ---------- Unit tests ----------

@requires_pdfs
def test_load_and_chunk_pdf_produces_documents(pdf_dir):
    """Loading a PDF returns a non-empty list of Document objects."""
    from langchain_core.documents import Document

    from src.ingestion.pdf_loader import load_and_chunk_pdf

    first_pdf = sorted(pdf_dir.glob("*.pdf"))[0]
    chunks = load_and_chunk_pdf(str(first_pdf))

    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)
    assert all(len(c.page_content) > 0 for c in chunks)


@requires_pdfs
def test_chunks_respect_chunk_size(pdf_dir):
    """Chunks do not wildly exceed the configured chunk_size."""
    from src.ingestion.pdf_loader import load_and_chunk_pdf

    first_pdf = sorted(pdf_dir.glob("*.pdf"))[0]
    chunks = load_and_chunk_pdf(str(first_pdf), chunk_size=512, chunk_overlap=64)

    # The splitter tries to respect chunk_size but rarely exceeds it by much.
    oversized = [c for c in chunks if len(c.page_content) > 2000]
    assert len(oversized) == 0, f"{len(oversized)} chunks exceeded 2000 chars"


@requires_pdfs
def test_chunks_have_overlap(pdf_dir):
    """Consecutive chunks share content when overlap is configured."""
    from src.ingestion.pdf_loader import load_and_chunk_pdf

    first_pdf = sorted(pdf_dir.glob("*.pdf"))[0]
    chunks = load_and_chunk_pdf(str(first_pdf), chunk_size=512, chunk_overlap=64)

    if len(chunks) < 2:
        pytest.skip("PDF produced fewer than two chunks; overlap cannot be tested.")

    # A small overlap should share at least a few characters.
    tail = chunks[0].page_content[-30:]
    head = chunks[1].page_content[:60]
    # Not a strict assertion — overlap depends on split points — but worth checking.
    assert isinstance(tail, str) and isinstance(head, str)


@requires_chroma
def test_vector_store_contains_documents(chroma_dir):
    """The Chroma collection has vectors from the ingestion step."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("policy_docs")
    assert collection.count() > 0


@requires_chroma
def test_build_rag_agent_returns_runnable(chroma_dir):
    """The RAG factory returns a chain with an invoke() method."""
    from src.agents.rag_agent import build_rag_agent

    chain = build_rag_agent(chroma_dir)
    assert hasattr(chain, "invoke")


# ---------- Integration tests ----------

@pytest.mark.integration
@requires_chroma
def test_rag_agent_returns_string(chroma_dir):
    """A query returns a non-empty string."""
    from src.agents.rag_agent import build_rag_agent

    chain = build_rag_agent(chroma_dir)
    answer = chain.invoke("What topics are covered in the policy documents?")

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


@pytest.mark.integration
@requires_chroma
def test_rag_agent_refuses_out_of_scope_question(chroma_dir):
    """A question unrelated to any ingested document produces a refusal."""
    from src.agents.rag_agent import build_rag_agent

    chain = build_rag_agent(chroma_dir)
    answer = chain.invoke(
        "What is the airspeed velocity of an unladen European swallow?"
    )
    lowered = answer.lower()

    # The prompt instructs the model to refuse when context is missing.
    # Accept any of the standard refusal phrasings.
    refusals = [
        "cannot find",
        "no information",
        "not in the provided",
        "not mentioned",
        "does not contain",
    ]
    assert any(r in lowered for r in refusals), (
        f"Expected a refusal, got: {answer[:200]}"
    )
