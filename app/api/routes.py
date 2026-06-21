from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, get_document_service, get_history_service
from app.core.config import get_settings
from app.models.schemas import HistoryResponse, QueryRequest, QueryResponse, UploadResponse
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.history_service import HistoryService

router = APIRouter()
MAX_QUERY_TOP_K = 10
MAX_MULTI_QUERY_COUNT = 5
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _is_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    return content_type == "application/pdf" or filename.endswith(".pdf")


async def _stream_ndjson(stream) -> AsyncIterator[str]:
    try:
        for item in stream:
            if item is None:
                continue

            yield str(item).rstrip("\n") + "\n"
    except Exception:
        yield json.dumps({"type": "error", "message": "Streaming interrupted; returning partial response."}) + "\n"
        return


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    try:
        if not _is_pdf_upload(file):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed. Please upload a valid .pdf file.")

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Uploaded file is too large.")

        payload = await document_service.ingest_upload(file)
        return UploadResponse(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document.") from exc


@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        settings = get_settings()
        top_k = max(1, min(request.top_k or settings.retriever_top_k, MAX_QUERY_TOP_K))
        multi_query_count = max(1, min(request.multi_query_count or settings.multi_query_count, MAX_MULTI_QUERY_COUNT))
        safe_request = request

        if safe_request.stream:
            stream = chat_service.stream_answer(safe_request, top_k=top_k, multi_query_count=multi_query_count)
            return StreamingResponse(_stream_ndjson(stream), media_type="application/x-ndjson")

        response = chat_service.answer(safe_request, top_k=top_k, multi_query_count=multi_query_count)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process query.") from exc


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    session_id: str = "default",
    history_service: HistoryService = Depends(get_history_service),
) -> HistoryResponse:
    try:
        return HistoryResponse(session_id=session_id, messages=history_service.get_history(session_id))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load history.") from exc
