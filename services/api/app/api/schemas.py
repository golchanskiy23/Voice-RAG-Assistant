import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.adapters.vector_stores.base import Retriever, StoreName
from app.services.rag import Latency
from app.services.search import SearchMode


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    chunks: int
    pages: int | None
    status: Literal["created", "replaced", "exists"]


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    mime: str
    pages: int | None
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    doc_ids: list[uuid.UUID] | None = None
    mode: SearchMode | None = None
    store: StoreName | None = None
    rerank: bool | None = None


class SearchChunkOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    ord: int
    text: str
    page: int | None
    section: str | None
    score: float
    source: Literal["vector", "bm25", "hybrid"]
    found_by: list[Retriever]


class SearchResponse(BaseModel):
    chunks: list[SearchChunkOut]
    store: StoreName
    mode: SearchMode
    reranked: bool
    latency: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class SourceOut(BaseModel):
    filename: str
    page: int | None
    section: str | None
    quote: str = Field(max_length=200)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    confidence: float = Field(ge=0, le=1)
    latency: Latency
