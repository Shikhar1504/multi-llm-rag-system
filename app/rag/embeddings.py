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
    api_key = getattr(settings, "openai_api_key", None)
    provider = settings.embeddings_provider
    model_name = settings.embeddings_model if provider != "openai" else settings.openai_embeddings_model

    try:
        return _cached_embeddings(provider, model_name, api_key)
    except Exception as exc:
        logger.warning("Embeddings provider %s failed, falling back to HuggingFace: %s", provider, exc)
        return _cached_embeddings("local", settings.embeddings_model, None)
