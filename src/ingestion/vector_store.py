"""Embed document chunks into the shared Chroma vector store."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.vectorstore import get_chroma_client


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
    )


def build_vector_store(chunks, persist_dir: str | None = None):
    """Create or append chunks to the policy_docs collection.

    Always appends. Uses the shared Chroma client so no singleton conflict
    can arise and so the RAG retriever sees new documents immediately.
    """
    # persist_dir is retained for signature compatibility; the shared
    # client already points at settings.chroma_persist_dir.
    vectorstore = Chroma(
        client=get_chroma_client(),
        collection_name=settings.chroma_collection_name,
        embedding_function=_embeddings(),
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def reset_vector_store() -> None:
    """Delete and recreate the collection. Used by --reset."""
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    client.create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
