import asyncio
from collections.abc import Awaitable, Callable, Sequence
from functools import cached_property

import httpx2
import openai
import tiktoken
from openai import AsyncOpenAI

from app.adapters.embedders.base import EMBEDDING_DIM
from app.core.retry import retry_async

RETRYABLE: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = EMBEDDING_DIM,
        batch_size: int = 128,
        attempts: int = 5,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0, http_client=http_client)
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._attempts = attempts
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return 8191

    @cached_property
    def _encoding(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(list(texts[start : start + self._batch_size])))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def is_ready(self) -> bool:
        return not self._client.is_closed()

    async def close(self) -> None:
        await self._client.close()

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async def request() -> list[list[float]]:
            response = await self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dim, encoding_format="float"
            )
            return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

        return await retry_async(
            request, retry_on=RETRYABLE, attempts=self._attempts, sleep=self._sleep
        )
