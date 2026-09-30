import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest
from gigachat import Chat, ChatCompletion, ChatCompletionChunk, MessagesRole, RateLimitError

from app.adapters.llm.base import (
    ChatMessage,
    Completion,
    LLMChunk,
    LLMClient,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.adapters.llm.fallback import FallbackLLMClient
from app.adapters.llm.gigachat import GigaChatLLMClient, to_chat
from app.adapters.llm.openai import OpenAILLMClient
from app.services.llm_answer import ANSWER_SCHEMA
from tests.llm_fakes import ScriptedLLM

REQUEST = LLMRequest(
    messages=(ChatMessage("system", "правила"), ChatMessage("user", "вопрос")),
    json_schema=ANSWER_SCHEMA,
    max_tokens=128,
)


def collect(stream: AsyncIterator[LLMChunk]) -> list[LLMChunk]:
    async def run() -> list[LLMChunk]:
        return [chunk async for chunk in stream]

    return asyncio.run(run())


def completion_body(content: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


def sse_body(deltas: list[str]) -> bytes:
    chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}],
        }
        for d in deltas
    ]
    chunks.append(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
    )
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


class FakeOpenAI:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [])
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.statuses and (status := self.statuses.pop(0)) != 200:
            return httpx2.Response(status, json={"error": {"message": "boom", "type": "x"}})
        if body.get("stream"):
            return httpx2.Response(
                200,
                content=sse_body(['{"answer": ', '"да"', "}"]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(200, json=completion_body('{"answer": "да"}'))

    def client(self) -> OpenAILLMClient:
        async def no_sleep(_: float) -> None:
            return None

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return OpenAILLMClient(api_key="test", http_client=http, sleep=no_sleep, attempts=3)


def test_openai_complete_sends_strict_schema_and_reads_usage() -> None:
    fake = FakeOpenAI()

    result = asyncio.run(fake.client().complete(REQUEST))

    assert result == Completion('{"answer": "да"}', "openai", Usage(11, 4))
    [body] = fake.requests
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["max_completion_tokens"] == 128
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA.schema


def test_openai_retries_rate_limit() -> None:
    fake = FakeOpenAI(statuses=[429, 503, 200])

    result = asyncio.run(fake.client().complete(REQUEST))

    assert result.text == '{"answer": "да"}'
    assert len(fake.requests) == 3


def test_openai_gives_up_with_llm_error() -> None:
    fake = FakeOpenAI(statuses=[500, 500, 500])

    with pytest.raises(LLMError, match="openai"):
        asyncio.run(fake.client().complete(REQUEST))


def test_openai_does_not_retry_bad_request() -> None:
    fake = FakeOpenAI(statuses=[400])

    with pytest.raises(LLMError):
        asyncio.run(fake.client().complete(REQUEST))
    assert len(fake.requests) == 1


def test_openai_stream_yields_deltas_and_usage() -> None:
    fake = FakeOpenAI()

    chunks = collect(fake.client().stream(REQUEST))

    assert chunks == [
        TextDelta('{"answer": '),
        TextDelta('"да"'),
        TextDelta("}"),
        StreamEnd("openai", Usage(11, 4)),
    ]
    assert fake.requests[0]["stream_options"] == {"include_usage": True}


GIGA_USAGE = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}


class FakeGigaChat:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.payloads: list[Chat] = []
        self.closed = False

    async def achat(self, payload: Chat) -> ChatCompletion:
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return ChatCompletion.model_validate(
            {
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "{}"}},
                ],
                "created": 1,
                "model": "GigaChat-2",
                "object": "chat.completion",
                "usage": GIGA_USAGE,
            }
        )

    async def astream(self, payload: Chat) -> AsyncIterator[ChatCompletionChunk]:
        self.payloads.append(payload)
        for i, text in enumerate(["{", "}"]):
            yield ChatCompletionChunk.model_validate(
                {
                    "choices": [{"index": 0, "delta": {"content": text}}],
                    "created": 1,
                    "model": "GigaChat-2",
                    "object": "chat.completion",
                    "usage": GIGA_USAGE if i == 1 else None,
                }
            )
        if self.error:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True


def test_to_chat_maps_roles_and_schema() -> None:
    chat = to_chat(REQUEST)
    payload = chat.model_dump(exclude_none=True, by_alias=True)

    assert [m.role for m in chat.messages] == [MessagesRole.SYSTEM, MessagesRole.USER]
    assert payload["max_tokens"] == 128
    assert "temperature" not in payload
    assert payload["response_format"] == {
        "type": "json_schema",
        "schema": ANSWER_SCHEMA.schema,
        "strict": True,
    }


def test_gigachat_complete_and_stream() -> None:
    fake = FakeGigaChat()
    client = GigaChatLLMClient(fake)

    assert asyncio.run(client.complete(REQUEST)) == Completion("{}", "gigachat", Usage(9, 3))
    assert collect(client.stream(REQUEST)) == [
        TextDelta("{"),
        TextDelta("}"),
        StreamEnd("gigachat", Usage(9, 3)),
    ]
    asyncio.run(client.close())
    assert fake.closed


def test_gigachat_errors_become_llm_error() -> None:
    error = RateLimitError("https://giga", 429, b"", None)
    client = GigaChatLLMClient(FakeGigaChat(error=error))

    with pytest.raises(LLMError, match="gigachat"):
        asyncio.run(client.complete(REQUEST))


class BrokenLLM(ScriptedLLM):
    @property
    def name(self) -> str:
        return "broken"

    async def complete(self, request: LLMRequest) -> Completion:
        raise LLMError(self.name, "down")


def fallback(primary: LLMClient) -> FallbackLLMClient:
    return FallbackLLMClient(primary, ScriptedLLM(["резерв", "резерв"]))


def test_fallback_uses_secondary_when_primary_fails() -> None:
    result = asyncio.run(fallback(BrokenLLM([])).complete(REQUEST))

    assert (result.text, result.provider) == ("резерв", "scripted")


def test_fallback_switches_stream_before_first_token() -> None:
    chunks = collect(fallback(ScriptedLLM(["основной"], fail_stream_after=0)).stream(REQUEST))

    assert "".join(c.text for c in chunks if isinstance(c, TextDelta)) == "резерв"
    assert chunks[-1] == StreamEnd("scripted", Usage(100, 20))


def test_fallback_does_not_switch_mid_stream() -> None:
    client = fallback(ScriptedLLM(["основной ответ"], fail_stream_after=3))

    with pytest.raises(LLMError):
        collect(client.stream(REQUEST))
