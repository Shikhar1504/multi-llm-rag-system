from datetime import datetime

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    status: str
    chunk_count: int
    skipped: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"
    stream: bool = True
    top_k: int | None = None
    multi_query_count: int | None = None
    rewrite_query: bool | None = None
    rerank: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    context_found: bool = True
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime
    sources: list[SourceCitation] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
