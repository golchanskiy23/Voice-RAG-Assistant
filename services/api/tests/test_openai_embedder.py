import asyncio
import json
from typing import Any

import httpx2
import openai
import pytest

from app.adapters.embedders.openai import OpenAIEmbedder


class FakeOpenAI:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [])
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx2.Response(status, json={"error": {"message": "boom", "type": "x"}})
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(text)), float(i)]}
            for i, text in enumerate(body["input"])
        ]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": list(reversed(data)),
                "model": body["model"],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def embedder(self, batch_size: int = 2, attempts: int = 3) -> OpenAIEmbedder:
        async def no_sleep(_: float) -> None:
            return None

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))
        return OpenAIEmbedder(
            api_key="test",
            dim=2,
            batch_size=batch_size,
            attempts=attempts,
            http_client=client,
            sleep=no_sleep,
        )


def test_documents_are_batched_and_order_is_preserved() -> None:
    fake = FakeOpenAI()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = asyncio.run(fake.embedder(batch_size=2).embed_documents(texts))

    assert [len(r["input"]) for r in fake.requests] == [2, 2, 1]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_request_asks_for_configured_dimension_and_model() -> None:
    fake = FakeOpenAI()

    asyncio.run(fake.embedder().embed_query("вопрос"))

    assert fake.requests[0]["dimensions"] == 2
    assert fake.requests[0]["model"] == "text-embedding-3-small"
    assert fake.requests[0]["encoding_format"] == "float"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_are_retried(status: int) -> None:
    fake = FakeOpenAI(statuses=[status, status, 200])

    vector = asyncio.run(fake.embedder(attempts=3).embed_query("abc"))

    assert vector == [3.0, 0.0]
    assert len(fake.requests) == 3


def test_retries_are_exhausted() -> None:
    fake = FakeOpenAI(statuses=[429, 429, 429])

    with pytest.raises(openai.RateLimitError):
        asyncio.run(fake.embedder(attempts=3).embed_query("abc"))
    assert len(fake.requests) == 3


@pytest.mark.parametrize(
    ("status", "error"), [(400, openai.BadRequestError), (401, openai.AuthenticationError)]
)
def test_client_errors_are_not_retried(status: int, error: type[Exception]) -> None:
    fake = FakeOpenAI(statuses=[status])

    with pytest.raises(error):
        asyncio.run(fake.embedder().embed_query("abc"))
    assert len(fake.requests) == 1


def test_tokens_are_counted_with_cl100k() -> None:
    embedder = FakeOpenAI().embedder()

    assert embedder.count_tokens("hello world") == 2
    assert embedder.max_tokens == 8191
