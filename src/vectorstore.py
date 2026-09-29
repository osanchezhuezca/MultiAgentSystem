"""Shared Chroma client.

Chroma enforces a process-level singleton keyed by persist directory. Any
two Chroma instances opened against the same path must share identical
settings, or Chroma raises ValueError. This module owns the single client
so every caller (RAG retrieval, PDF ingestion, the CLI, tests) reuses it.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return the process-wide Chroma client.

    Cached so repeated calls return the same instance. The first call
    establishes the canonical settings; every later call reuses them.
    """
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(
            is_persistent=True,
            persist_directory=settings.chroma_persist_dir,
            anonymized_telemetry=False,
        ),
    )


def get_collection():
    """Return the policy_docs collection from the shared client."""
    return get_chroma_client().get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_client_cache() -> None:
    """Drop the cached client. Useful in tests that rebuild the store."""
    get_chroma_client.cache_clear()
