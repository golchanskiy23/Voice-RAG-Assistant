import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, Protocol

from app.api.schemas import SourceOut
from app.services.rag import FinalEvent, StreamEvent, TokenEvent

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ClientConnection(Protocol):
    async def is_disconnected(self) -> bool: ...


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def encode_event(event: StreamEvent) -> list[str]:
    match event:
        case TokenEvent(text=text):
            return [format_sse("token", {"text": text})]
        case FinalEvent(sources=sources, confidence=confidence, latency_ms=latency):
            payload = [
                SourceOut(filename=s.filename, page=s.page, section=s.section, quote=s.quote)
                for s in sources
            ]
            return [
                format_sse(
                    "sources",
                    {
                        "sources": [s.model_dump() for s in payload],
                        "confidence": confidence,
                    },
                ),
                format_sse("done", {"latency": latency}),
            ]


async def stream_sse(
    connection: ClientConnection,
    events: AsyncGenerator[StreamEvent],
    request_id: str | None,
) -> AsyncIterator[str]:
    async with aclosing(events) as stream:
        try:
            async for event in stream:
                if await connection.is_disconnected():
                    return
                for chunk in encode_event(event):
                    yield chunk
        except Exception:
            yield format_sse(
                "error",
                {
                    "error": {
                        "code": "stream_error",
                        "message": "Ошибка при генерации ответа",
                        "request_id": request_id,
                    }
                },
            )
