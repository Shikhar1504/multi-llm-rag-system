from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, Iterator

from langchain_core.documents import Document

from app.core.config import Settings
from app.core.llm import invoke_text, stream_text
from app.models.schemas import SourceCitation
from app.rag.vector_stores.base import VectorStoreBackend


DEFAULT_SYSTEM_PROMPT = (
    "You are a production-grade RAG assistant. "
    "Use only the provided context. "
    "If the answer is not supported by the context, say you could not find it in the uploaded documents. "
    "Always be concise and include source references when possible."
)


@dataclass(slots=True)
class RetrievedContext:
    rewritten_query: str
    documents: list[Document]
    sources: list[SourceCitation]
    context_text: str
    used_rewrite: bool = False
    used_multi_query: bool = False
    used_rerank: bool = False
    context_truncated: bool = False
    retrieval_mode: str = "direct"


class RAGPipeline:
    def __init__(self, *, settings: Settings, vector_store: VectorStoreBackend, llm) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._llm = llm
        self._logger = logging.getLogger(__name__)
        self._reranker = self._load_reranker()

    def _load_reranker(self):
        if not self._settings.enable_reranking:
            return None

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - optional dependency
            self._logger.warning("Reranker unavailable; continuing without reranking: %s", exc)
            return None

        try:
            return CrossEncoder(self._settings.reranker_model_name)
        except Exception as exc:  # pragma: no cover - model load failure is runtime/environment specific
            self._logger.exception("Failed to load reranker '%s': %s", self._settings.reranker_model_name, exc)
            return None

    def _is_simple_query(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return True

        word_count = len(normalized.split())
        if word_count > self._settings.simple_query_max_words:
            return False
        if len(normalized) > self._settings.simple_query_max_chars:
            return False

        complex_markers = (
            " compare ",
            " difference ",
            " versus ",
            " vs ",
            " summarize ",
            " explain ",
            " step by step ",
            " why ",
            " how does ",
            " pros and cons ",
        )
        padded = f" {normalized} "
        return not any(marker in padded for marker in complex_markers)

    def _should_expand_query(self, query: str) -> bool:
        if self._is_simple_query(query):
            return False
        return self._settings.enable_query_rewrite or self._settings.enable_multi_query

    def _rewrite_query(self, query: str) -> str:
        if not self._settings.enable_query_rewrite:
            return query

        prompt = (
            "Rewrite the user question into a clean search query for document retrieval. "
            "Preserve domain-specific terms and do not add explanations.\n\n"
            f"Question: {query}"
        )
        rewritten = invoke_text(self._llm, "You rewrite search queries.", prompt).strip()
        return rewritten or query

    def _generate_variants(self, query: str) -> list[str]:
        if not self._settings.enable_multi_query:
            return [query]

        prompt = (
            f"Generate {self._settings.multi_query_count} alternate search queries for the same user question. "
            "Return one query per line without numbering.\n\n"
            f"Question: {query}"
        )
        raw = invoke_text(self._llm, "You expand search queries.", prompt)
        variants = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
        variants = [variant for variant in variants if variant]
        if query not in variants:
            variants.insert(0, query)
        return variants[: max(1, self._settings.multi_query_count)]

    @staticmethod
    def _dedupe_documents(documents: Iterable[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for document in documents:
            metadata = document.metadata or {}
            content = document.page_content[:512].strip().lower()
            fingerprint_input = "|".join(
                [
                    str(metadata.get("source", "")),
                    str(metadata.get("page", metadata.get("page_number", ""))),
                    str(metadata.get("chunk_index", "")),
                    content,
                ]
            )
            key = hashlib.blake2s(fingerprint_input.encode("utf-8"), digest_size=12).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        return unique

    def _should_rerank(self, query: str, documents: list[Document], desired_top_k: int, used_expansion: bool) -> bool:
        if not self._reranker or not documents:
            return False

        if len(documents) == 1:
            return False

        if len(documents) < desired_top_k:
            return True

        if used_expansion:
            return True

        source_count = len({str((document.metadata or {}).get("source", "")) for document in documents if (document.metadata or {}).get("source")})
        return source_count <= 1 and len(query.split()) > self._settings.simple_query_max_words

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not self._reranker or not documents:
            return documents

        scores = self._reranker.predict([[query, doc.page_content] for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
        return [document for document, _ in ranked[: self._settings.rerank_top_k]]

    def _truncate_context(self, documents: list[Document]) -> tuple[list[Document], str, bool]:
        if not documents:
            return [], "", False

        context_parts: list[str] = []
        truncated = False
        remaining = self._settings.max_context_chars

        for index, document in enumerate(documents):
            metadata = document.metadata or {}
            source = SourceCitation(
                source=str(metadata.get("source", "unknown")),
                page=metadata.get("page") or metadata.get("page_number"),
                chunk_id=str(metadata.get("chunk_index", index)),
            )

            header = f"[Source {index + 1}] {source.source} page={source.page if source.page is not None else 'n/a'}\n"
            body = document.page_content.strip()
            separator_cost = 2 if context_parts else 0
            available = remaining - separator_cost

            if available <= len(header):
                truncated = True
                break

            body_budget = available - len(header)
            if len(body) > body_budget:
                body = body[: max(0, body_budget - 1)].rstrip() + "…"
                truncated = True

            part = f"{header}{body}"
            context_parts.append(part)
            remaining -= len(part) + separator_cost

            if remaining <= 0:
                truncated = True
                break

        return documents[: len(context_parts)], "\n\n".join(context_parts), truncated

    def _build_retrieval_context(
        self,
        *,
        query: str,
        documents: list[Document],
        rewritten_query: str,
        used_rewrite: bool,
        used_multi_query: bool,
        used_rerank: bool,
        retrieval_mode: str,
    ) -> RetrievedContext:
        truncated_documents, context_text, context_truncated = self._truncate_context(documents)
        sources: list[SourceCitation] = []
        for index, document in enumerate(truncated_documents):
            metadata = document.metadata or {}
            sources.append(
                SourceCitation(
                    source=str(metadata.get("source", "unknown")),
                    page=metadata.get("page") or metadata.get("page_number"),
                    chunk_id=str(metadata.get("chunk_index", index)),
                )
            )

        return RetrievedContext(
            rewritten_query=rewritten_query,
            documents=truncated_documents,
            sources=sources,
            context_text=context_text,
            used_rewrite=used_rewrite,
            used_multi_query=used_multi_query,
            used_rerank=used_rerank,
            context_truncated=context_truncated,
            retrieval_mode=retrieval_mode,
        )

    def _build_answer_prompt(self, context_text: str, query: str) -> str:
        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context. If the context is insufficient, say you could not find the answer in the uploaded documents. "
            "Include brief source attribution in the response."
        )

    @staticmethod
    def _finalize_answer(answer: str) -> str:
        cleaned = answer.strip()
        return cleaned or "I could not find the answer in the uploaded documents."

    def retrieve(self, query: str, *, top_k: int | None = None, fetch_k: int | None = None) -> RetrievedContext:
        effective_top_k = top_k or self._settings.retriever_top_k
        effective_fetch_k = fetch_k or self._settings.retriever_fetch_k
        use_expansion = self._should_expand_query(query)

        rewritten_query = query
        used_rewrite = False
        used_multi_query = False
        retrieval_mode = "direct"

        if use_expansion and self._settings.enable_query_rewrite:
            rewritten_query = self._rewrite_query(query)
            used_rewrite = rewritten_query != query

        variants = [rewritten_query]
        if use_expansion and self._settings.enable_multi_query:
            variants = self._generate_variants(rewritten_query)
            used_multi_query = len(variants) > 1
            retrieval_mode = "expanded"

        retrieved: list[Document] = []
        for variant in variants:
            retrieved.extend(self._vector_store.search(variant, effective_top_k, effective_fetch_k))

        documents = self._dedupe_documents(retrieved)
        used_rerank = self._should_rerank(query, documents, effective_top_k, use_expansion)
        if used_rerank:
            documents = self._rerank(rewritten_query, documents)
            retrieval_mode = f"{retrieval_mode}+reranked" if retrieval_mode != "direct" else "reranked"

        return self._build_retrieval_context(
            query=query,
            documents=documents,
            rewritten_query=rewritten_query,
            used_rewrite=used_rewrite,
            used_multi_query=used_multi_query,
            used_rerank=used_rerank,
            retrieval_mode=retrieval_mode,
        )

    def answer(self, query: str, *, top_k: int | None = None, fetch_k: int | None = None) -> tuple[str, RetrievedContext, bool]:
        context = self.retrieve(query, top_k=top_k, fetch_k=fetch_k)
        if not context.documents:
            return ("I could not find the answer in the uploaded documents.", context, False)

        prompt = self._build_answer_prompt(context.context_text, query)
        answer = self._finalize_answer(invoke_text(self._llm, DEFAULT_SYSTEM_PROMPT, prompt))
        if answer == "I could not find the answer in the uploaded documents.":
            return answer, context, False
        return answer, context, True

    def stream_answer(self, query: str, *, top_k: int | None = None, fetch_k: int | None = None) -> tuple[RetrievedContext, Iterable[str], bool]:
        context = self.retrieve(query, top_k=top_k, fetch_k=fetch_k)
        if not context.documents:
            return context, ["I could not find the answer in the uploaded documents."], False

        prompt = self._build_answer_prompt(context.context_text, query)
        return context, stream_text(self._llm, DEFAULT_SYSTEM_PROMPT, prompt), True
