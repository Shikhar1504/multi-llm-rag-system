from __future__ import annotations

import json
import logging

from app.core.config import Settings
from app.models.schemas import QueryRequest, QueryResponse, SourceCitation
from app.rag.pipeline import RAGPipeline
from app.services.history_service import HistoryService


class ChatService:
    def __init__(self, *, settings: Settings, pipeline: RAGPipeline, history: HistoryService) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._history = history
        self._logger = logging.getLogger(__name__)

    def answer(self, request: QueryRequest, *, top_k: int | None = None, multi_query_count: int | None = None) -> QueryResponse:
        self._history.add_message(request.session_id, "user", request.question)

        pipeline_kwargs = {
            "top_k": top_k if top_k is not None else request.top_k,
            "multi_query_count": multi_query_count if multi_query_count is not None else request.multi_query_count,
            "fetch_k": self._settings.retriever_fetch_k,
        }

        try:
            answer, context, context_found = self._pipeline.answer(request.question, **pipeline_kwargs)
        except TypeError:
            pipeline_kwargs.pop("multi_query_count", None)
            answer, context, context_found = self._pipeline.answer(request.question, **pipeline_kwargs)

        sources = context.sources
        self._history.add_message(request.session_id, "assistant", answer, sources=sources)

        return QueryResponse(
            answer=answer,
            rewritten_query=context.rewritten_query,
            sources=sources,
            context_found=context_found,
            session_id=request.session_id,
        )

    def stream_answer(self, request: QueryRequest, *, top_k: int | None = None, multi_query_count: int | None = None):
        self._history.add_message(request.session_id, "user", request.question)

        pipeline_kwargs = {
            "top_k": top_k if top_k is not None else request.top_k,
            "multi_query_count": multi_query_count if multi_query_count is not None else request.multi_query_count,
            "fetch_k": self._settings.retriever_fetch_k,
        }

        try:
            context, stream, context_found = self._pipeline.stream_answer(request.question, **pipeline_kwargs)
        except TypeError:
            pipeline_kwargs.pop("multi_query_count", None)
            context, stream, context_found = self._pipeline.stream_answer(request.question, **pipeline_kwargs)

        def generator():
            payload = {
                "type": "sources",
                "rewritten_query": context.rewritten_query,
                "context_found": context_found,
                "retrieval_mode": context.retrieval_mode,
                "context_truncated": context.context_truncated,
                "sources": [source.model_dump() for source in context.sources],
            }
            yield json.dumps(payload) + "\n"

            answer = ""
            try:
                for piece in stream:
                    if not piece:
                        continue
                    answer += piece
                    yield json.dumps({"type": "token", "content": piece}) + "\n"
                answer = answer.strip() or "I could not find the answer in the uploaded documents."
            except Exception as exc:  # pragma: no cover - streaming failures are runtime-specific
                self._logger.exception("Streaming failed for session %s: %s", request.session_id, exc)
                answer = answer.strip() or "I could not find the answer in the uploaded documents."
                yield json.dumps({"type": "error", "message": "Stream interrupted; returning partial answer."}) + "\n"
            finally:
                self._history.add_message(request.session_id, "assistant", answer, sources=context.sources)
                yield json.dumps({"type": "done", "answer": answer}) + "\n"

        return generator()
