from typing import Literal

from pydantic import BaseModel, Field

from app.services.rag import Latency


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


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
