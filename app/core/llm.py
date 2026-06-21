from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections.abc import Iterable
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings

FALLBACK_RESPONSE = "I could not generate a response at this time."
LLM_TIMEOUT_SECONDS = 30
logger = logging.getLogger(__name__)
# Reused globally so invoke/stream calls do not create new workers each time.
# Application shutdown should handle executor lifecycle cleanup.
_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _build_messages(system_prompt: str, human_prompt: str) -> list[SystemMessage | HumanMessage]:
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _build_prompt_text(system_prompt: str, human_prompt: str) -> str:
    return f"{system_prompt}\n\n{human_prompt}".strip()


def _response_to_text(response) -> str:
    if response is None:
        return FALLBACK_RESPONSE

    content = getattr(response, "content", response)
    if content is None:
        return FALLBACK_RESPONSE

    text = str(content).strip()
    return text or FALLBACK_RESPONSE


def _run_with_timeout(callable_fn, *, fallback: str = FALLBACK_RESPONSE) -> str:
    future = _LLM_EXECUTOR.submit(callable_fn)
    try:
        result = future.result(timeout=LLM_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        future.cancel()
        return fallback
    except Exception:
        return fallback

    if result is None:
        return fallback

    return str(result).strip() or fallback


def _start_stream_with_timeout(callable_fn):
    future = _LLM_EXECUTOR.submit(callable_fn)
    try:
        return future.result(timeout=LLM_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        future.cancel()
        return None
    except Exception:
        return None


@lru_cache(maxsize=8)
def _cached_chat_model(
    provider: str,
    model_name: str,
    temperature: float,
    gemini_api_key: str | None,
    mistral_api_key: str | None,
    huggingface_api_key: str | None,
    openai_api_key: str | None,
):
    return _build_chat_model(
        provider,
        model_name,
        temperature,
        gemini_api_key=gemini_api_key,
        mistral_api_key=mistral_api_key,
        huggingface_api_key=huggingface_api_key,
        openai_api_key=openai_api_key,
    )


def _fallback_chat_model(
    model_name: str,
    temperature: float,
    *,
    gemini_api_key: str | None,
    mistral_api_key: str | None,
    huggingface_api_key: str | None,
    openai_api_key: str | None,
):
    for provider in ("mistral", "huggingface"):
        try:
            return _build_chat_model(
                provider,
                model_name,
                temperature,
                gemini_api_key=gemini_api_key,
                mistral_api_key=mistral_api_key,
                huggingface_api_key=huggingface_api_key,
                openai_api_key=openai_api_key,
            )
        except Exception as exc:
            logger.warning("LLM fallback provider %s failed: %s", provider, exc)

    raise RuntimeError("Failed to initialize a fallback LLM provider")


def _build_chat_model(
    provider: str,
    model_name: str,
    temperature: float,
    *,
    gemini_api_key: str | None,
    mistral_api_key: str | None,
    huggingface_api_key: str | None,
    openai_api_key: str | None,
):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=gemini_api_key)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model_name, temperature=temperature, mistral_api_key=mistral_api_key)

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEndpoint

            return HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=huggingface_api_key,
                task="text-generation",
                temperature=temperature,
            )
        except Exception:
            from langchain_community.llms import HuggingFacePipeline

            try:
                from transformers import pipeline as hf_pipeline
            except Exception as exc:
                raise RuntimeError("HuggingFace local model support requires transformers") from exc

            generator = hf_pipeline("text-generation", model=model_name)
            return HuggingFacePipeline(pipeline=generator)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_chat_model(settings: Settings):
    try:
        return _cached_chat_model(
            settings.llm_provider,
            settings.llm_model,
            settings.temperature,
            settings.gemini_api_key,
            settings.mistral_api_key,
            settings.huggingface_api_key,
            settings.openai_api_key,
        )
    except Exception as exc:
        logger.warning("LLM provider %s failed, falling back to Mistral/HuggingFace: %s", settings.llm_provider, exc)
        return _fallback_chat_model(
            settings.llm_model,
            settings.temperature,
            gemini_api_key=settings.gemini_api_key,
            mistral_api_key=settings.mistral_api_key,
            huggingface_api_key=settings.huggingface_api_key,
            openai_api_key=settings.openai_api_key,
        )


def invoke_text(model, system_prompt: str, human_prompt: str) -> str:
    messages = _build_messages(system_prompt, human_prompt)
    prompt_text = _build_prompt_text(system_prompt, human_prompt)

    def call() -> str:
        try:
            response = model.invoke(messages)
        except (TypeError, ValueError, AttributeError):
            response = model.invoke(prompt_text)
        return _response_to_text(response)

    return _run_with_timeout(call)


def stream_text(model, system_prompt: str, human_prompt: str) -> Iterable[str]:
    def fallback_once() -> Iterable[str]:
        yield FALLBACK_RESPONSE
        return

    messages = _build_messages(system_prompt, human_prompt)
    prompt_text = _build_prompt_text(system_prompt, human_prompt)

    def open_stream():
        try:
            return model.stream(messages)
        except (TypeError, ValueError, AttributeError):
            return model.stream(prompt_text)

    stream = _start_stream_with_timeout(open_stream)
    if stream is None:
        yield from fallback_once()
        return

    try:
        for chunk in stream:
            try:
                content = getattr(chunk, "content", None)
                if content is None:
                    continue

                yield str(content)
            except Exception:
                continue
    except Exception:
        yield from fallback_once()
        return
