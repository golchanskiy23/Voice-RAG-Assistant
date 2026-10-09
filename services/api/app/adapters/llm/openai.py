import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx2
import openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from app.adapters.embedders.openai import RETRYABLE
from app.adapters.llm.base import (
    Completion,
    LLMChunk,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.core.retry import retry_async


def to_openai_messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    for message in request.messages:
        match message.role:
            case "system":
                messages.append({"role": "system", "content": message.content})
            case "user":
                messages.append({"role": "user", "content": message.content})
            case "assistant":
                messages.append({"role": "assistant", "content": message.content})
    return messages


def response_format(request: LLMRequest) -> ResponseFormatJSONSchema | openai.Omit:
    if request.json_schema is None:
        return openai.omit
    return {
        "type": "json_schema",
        "json_schema": {
            "name": request.json_schema.name,
            "schema": request.json_schema.schema,
            "strict": True,
        },
    }


class OpenAILLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        attempts: int = 3,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key, max_retries=0, timeout=timeout, http_client=http_client
        )
        self._model = model
        self._attempts = attempts
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, request: LLMRequest) -> Completion:
        async def create() -> Completion:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=to_openai_messages(request),
                response_format=response_format(request),
                max_completion_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            usage = response.usage
            return Completion(
                text=response.choices[0].message.content or "",
                provider=self.name,
                usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else None,
            )

        try:
            return await self._retry(create)
        except openai.OpenAIError as exc:
            raise LLMError(self.name, type(exc).__name__) from exc

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        async def open_stream() -> AsyncStream[ChatCompletionChunk]:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=to_openai_messages(request),
                response_format=response_format(request),
                max_completion_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

        usage: Usage | None = None
        try:
            stream = await self._retry(open_stream)
            async with stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                    for choice in chunk.choices:
                        if choice.delta.content:
                            yield TextDelta(choice.delta.content)
        except openai.OpenAIError as exc:
            raise LLMError(self.name, type(exc).__name__) from exc
        yield StreamEnd(provider=self.name, usage=usage)

    async def is_ready(self) -> bool:
        return not self._client.is_closed()

    async def close(self) -> None:
        await self._client.close()

    async def _retry[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        return await retry_async(
            operation, retry_on=RETRYABLE, attempts=self._attempts, sleep=self._sleep
        )
