from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _cached_embeddings(provider: str, model_name: str, api_key: str | None):
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if api_key:
            return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
        return OpenAIEmbeddings(model=model_name)

    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(f"Unsupported embeddings provider: {provider}")


def build_embeddings(settings: Settings):
    provider = settings.embeddings_provider
    if provider == "gemini" and not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when EMBEDDINGS_PROVIDER=gemini")

    if provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")

    api_key = settings.gemini_api_key if provider == "gemini" else settings.openai_api_key if provider == "openai" else None
    model_name = settings.embeddings_model if provider != "openai" else settings.openai_embeddings_model

    try:
        return _cached_embeddings(provider, model_name, api_key)
    except Exception as exc:
        logger.warning("Embeddings provider %s failed, falling back to HuggingFace: %s", provider, exc)
        return _cached_embeddings("local", settings.embeddings_model, None)
