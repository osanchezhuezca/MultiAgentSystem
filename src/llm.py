"""LLM factory — returns the right chat model based on settings.llm_provider."""
from __future__ import annotations

from typing import Any, Optional

from src.config import settings


def get_llm(model: Optional[str] = None, **overrides: Any):
    provider = settings.llm_provider.lower()

    params: dict[str, Any] = {
        "temperature": overrides.pop("temperature", settings.llm_temperature),
    }

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Add it to .env or export it in your shell."
            )
        return ChatOpenAI(
            model=model or settings.openai_llm_model,
            api_key=settings.openai_api_key,          # ← the critical fix
            max_tokens=overrides.pop("max_tokens", settings.llm_max_tokens),
            **params,
            **overrides,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is empty. Add it to .env.")
        return ChatGroq(
            model=model or settings.groq_llm_model,
            api_key=settings.groq_api_key,            # ← same pattern
            **params,
            **overrides,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model or settings.ollama_llm_model,
            **params,
            **overrides,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def get_router_llm(**overrides: Any):
    """Cheap, fast model used by the supervisor router to classify intent."""
    provider = settings.llm_provider.lower()

    if provider == "openai":
        return get_llm(model=settings.openai_router_model, **overrides)

    if provider == "groq":
        return get_llm(model=settings.groq_router_model, **overrides)

    # Ollama: use the default model
    return get_llm(**overrides)