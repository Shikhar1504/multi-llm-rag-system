from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.llm import build_chat_model
from app.rag.pipeline import RAGPipeline
from app.rag.vector_stores.factory import build_vector_store
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.history_service import HistoryService


@lru_cache(maxsize=1)
def get_vector_store():
    settings = get_settings()
    try:
        return build_vector_store(settings)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize vector store backend '{settings.vector_store_backend}'") from exc


@lru_cache(maxsize=1)
def get_history_service() -> HistoryService:
    settings = get_settings()
    try:
        return HistoryService(settings.history_db_path)
    except Exception as exc:
        raise RuntimeError("Failed to initialize history service") from exc


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    try:
        vector_store = get_vector_store()
        llm = build_chat_model(settings)
        return RAGPipeline(settings=settings, vector_store=vector_store, llm=llm)
    except Exception as exc:
        raise RuntimeError("Failed to initialize RAG pipeline") from exc


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    settings = get_settings()
    try:
        return DocumentService(settings=settings, vector_store=get_vector_store())
    except Exception as exc:
        raise RuntimeError("Failed to initialize document service") from exc


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    try:
        return ChatService(settings=settings, pipeline=get_pipeline(), history=get_history_service())
    except Exception as exc:
        raise RuntimeError("Failed to initialize chat service") from exc
