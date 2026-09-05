from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every variable can be overridden with a DOCQA_ env var."""

    model_config = SettingsConfigDict(
        env_prefix="DOCQA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Postgres / pgvector ---
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/docqa"
    collection_name: str = "policies"

    # --- Embeddings ---
    # "stub" (deterministic, tests) | "openai" (OpenAI-compatible) | "local" (sentence-transformers)
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    openai_api_base: str = ""  # e.g. https://api.openai.com/v1 or http://ollama:11434/v1
    openai_api_key: str = ""

    # --- LLM ---
    # "openai_compat" (OpenAI/Groq/Ollama/vLLM via base URL) | "stub"
    llm_provider: str = "openai_compat"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # --- Retrieval ---
    top_k: int = 4
    vector_k: int = 8
    keyword_k: int = 8
    reranker: str = "none"  # "none" | "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Ingestion ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- LangGraph ---
    checkpointer_path: str = "checkpoints.db"

    # --- Tracing (optional) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
