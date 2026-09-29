"""Centralized configuration loaded from .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    # ---------- LLM provider keys ----------
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    huggingfacehub_api_token: str = ""

    # ---------- LLM config ----------
    llm_provider: str = "openai"
    openai_llm_model: str = "gpt-4o"
    openai_router_model: str = "gpt-4o-mini"
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_router_model: str = "openai/gpt-oss-20b"
    ollama_llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---------- Embeddings ----------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True

    # ---------- Database ----------
    database_url: str = "sqlite:///data/processed/customer_support.db"

    # ---------- Vector store ----------
    chroma_persist_dir: str = "data/processed/chroma_db"
    chroma_collection_name: str = "policy_docs"

    # ---------- Ingestion ----------
    pdf_chunk_size: int = 512
    pdf_chunk_overlap: int = 64
    rag_top_k: int = 4

    # ---------- Data paths ----------
    raw_data_dir: str = "data/raw"
    policy_pdf_dir: str = "data/raw/policy_documents"
    tickets_csv: str = "data/raw/customer_support_tickets.csv"

    # ---------- MCP ----------
    mcp_server_name: str = "CustomerSupportAgents"
    mcp_transport: str = "stdio"

    # ---------- UI ----------
    streamlit_page_title: str = "Multi-Agent Customer Support"
    streamlit_port: int = 8501

    # ---------- Logging ----------
    log_level: str = "INFO"
    langchain_tracing_v2: bool = False
    langchain_project: str = "tcs-multi-agent"
    langchain_api_key: str = ""

    # ---------- App ----------
    app_env: str = "development"


settings = Settings()
