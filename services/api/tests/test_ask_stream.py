import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from app.adapters.llm.base import LLMChunk, LLMRequest, TextDelta
from app.adapters.llm.stub import StubLLMClient
from app.api.sse import stream_sse
from app.services.rag import FinalEvent, StreamEvent, TokenEvent
from tests.factories import build_app
from tests.test_search import FAQ, upload


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_stream_emits_tokens_then_sources_then_done(client: TestClient) -> None:
    upload(client, "faq.md", FAQ)
    with client.stream(
        "POST", "/ask/stream", json={"question": "какие файлы отклоняются"}
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    events = parse_sse(body)
    names = [name for name, _ in events]
    tokens = [data["text"] for name, data in events if name == "token"]
    assert names[-2:] == ["sources", "done"]
    assert set(names[:-2]) == {"token"}
    assert len(tokens) > 1
    assert "".join(tokens) == "Файлы больше двадцати мегабайт отклоняются."
    sources = events[-2][1]
    assert [s["filename"] for s in sources["sources"]] == ["faq.md"]
    assert sources["confidence"] == 0.5
    assert {"first_token", "llm", "total"} <= set(events[-1][1]["latency"])


def test_stream_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask/stream", json={"question": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


class FailingLLM(StubLLMClient):
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        yield TextDelta('{"answer": "Начало ')
        raise RuntimeError("provider down")


def test_stream_reports_error_event_mid_stream() -> None:
    with TestClient(build_app(llm_factory=FailingLLM)) as client:
        upload(client, "faq.md", FAQ)
        with client.stream(
            "POST",
            "/ask/stream",
            json={"question": "файлы больше"},
            headers={"X-Request-ID": "req-9"},
        ) as response:
            events = parse_sse(response.read().decode())

    assert events[0] == ("token", {"text": "Начало "})
    name, data = events[-1]
    assert name == "error"
    assert data["error"]["code"] == "stream_error"
    assert data["error"]["request_id"] == "req-9"
    assert "provider down" not in data["error"]["message"]


class DisconnectAfter:
    def __init__(self, checks: int) -> None:
        self._remaining = checks

    async def is_disconnected(self) -> bool:
        self._remaining -= 1
        return self._remaining < 0


def test_stream_stops_and_closes_generator_on_disconnect() -> None:
    produced: list[str] = []
    closed = asyncio.Event()

    async def events() -> AsyncGenerator[StreamEvent]:
        try:
            for token in ["a", "b", "c", "d"]:
                produced.append(token)
                yield TokenEvent(token)
            yield FinalEvent(sources=[], confidence=0.0, latency_ms={})
        finally:
            closed.set()

    async def collect() -> list[str]:
        return [chunk async for chunk in stream_sse(DisconnectAfter(2), events(), None)]

    chunks = asyncio.run(collect())

    assert len(chunks) == 2
    assert produced == ["a", "b", "c"]
    assert closed.is_set()
