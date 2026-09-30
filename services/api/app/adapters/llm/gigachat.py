from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from gigachat import (
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    GigaChatAsyncClient,
    GigaChatException,
    Messages,
    MessagesRole,
)

from app.adapters.llm.base import (
    Completion,
    LLMChunk,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)

ROLES: dict[str, MessagesRole] = {
    "system": MessagesRole.SYSTEM,
    "user": MessagesRole.USER,
    "assistant": MessagesRole.ASSISTANT,
}


class GigaChatTransport(Protocol):
    async def achat(self, payload: Chat) -> ChatCompletion: ...

    def astream(self, payload: Chat) -> AsyncIterator[ChatCompletionChunk]: ...

    async def aclose(self) -> None: ...


class SdkTransport:
    def __init__(self, client: GigaChatAsyncClient) -> None:
        self._client = client

    async def achat(self, payload: Chat) -> ChatCompletion:
        return await self._client.achat(payload)

    def astream(self, payload: Chat) -> AsyncIterator[ChatCompletionChunk]:
        return self._client.astream(payload)

    async def aclose(self) -> None:
        await self._client.aclose()


def to_chat(request: LLMRequest, stream: bool = False) -> Chat:
    response_format: dict[str, Any] | None = None
    if request.json_schema is not None:
        response_format = {
            "type": "json_schema",
            "schema": request.json_schema.schema,
            "strict": True,
        }
    return Chat(
        messages=[Messages(role=ROLES[m.role], content=m.content) for m in request.messages],
        max_tokens=request.max_tokens,
        temperature=request.temperature or None,
        response_format=response_format,
        stream=stream or None,
    )


class GigaChatLLMClient:
    def __init__(self, transport: GigaChatTransport) -> None:
        self._transport = transport
        self._closed = False

    @classmethod
    def from_credentials(
        cls,
        credentials: str,
        scope: str,
        model: str,
        timeout: float = 30.0,
        verify_ssl_certs: bool = True,
        ca_bundle_file: str | None = None,
        attempts: int = 3,
    ) -> "GigaChatLLMClient":
        client = GigaChatAsyncClient(
            credentials=credentials,
            scope=scope,
            model=model,
            timeout=timeout,
            verify_ssl_certs=verify_ssl_certs,
            ca_bundle_file=ca_bundle_file,
            max_retries=attempts - 1,
        )
        return cls(SdkTransport(client))

    @property
    def name(self) -> str:
        return "gigachat"

    async def complete(self, request: LLMRequest) -> Completion:
        try:
            response = await self._transport.achat(to_chat(request))
        except (GigaChatException, httpx.HTTPError) as exc:
            raise LLMError(self.name, type(exc).__name__) from exc
        return Completion(
            text=response.choices[0].message.content,
            provider=self.name,
            usage=Usage(response.usage.prompt_tokens, response.usage.completion_tokens),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        usage: Usage | None = None
        try:
            async for chunk in self._transport.astream(to_chat(request, stream=True)):
                if chunk.usage is not None:
                    usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                for choice in chunk.choices:
                    if choice.delta.content:
                        yield TextDelta(choice.delta.content)
        except (GigaChatException, httpx.HTTPError) as exc:
            raise LLMError(self.name, type(exc).__name__) from exc
        yield StreamEnd(provider=self.name, usage=usage)

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
        await self._transport.aclose()
