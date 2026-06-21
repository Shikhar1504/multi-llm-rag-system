from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Production RAG Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    api_prefix: str = "/api"

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    history_db_path: Path = Path("data/history.sqlite3")
    registry_db_path: Path = Path("data/registry.sqlite3")

    vector_store_backend: Literal["chroma", "pinecone"] = "chroma"
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "rag-documents"
    pinecone_namespace: str = "default"

    gemini_api_key: str | None = None
    mistral_api_key: str | None = None
    huggingface_api_key: str | None = None
    openai_api_key: str | None = None

    embeddings_provider: Literal["local", "gemini", "openai"] = "local"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embeddings_model: str = "text-embedding-3-small"

    llm_provider: Literal["gemini", "mistral", "huggingface", "openai"] = "mistral"
    llm_model: str = "mistral-small-latest"
    temperature: float = 0.0

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 4
    retriever_fetch_k: int = 12
    multi_query_count: int = 3
    rerank_top_k: int = 4
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    simple_query_max_words: int = 8
    simple_query_max_chars: int = 90
    max_context_chars: int = 8000

    enable_multi_query: bool = True
    enable_reranking: bool = True
    enable_query_rewrite: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
