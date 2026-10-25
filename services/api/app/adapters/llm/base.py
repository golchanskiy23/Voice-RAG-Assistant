from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.adapters.base import Resource

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class JsonSchema:
    name: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[ChatMessage, ...]
    json_schema: JsonSchema | None = None
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    usage: Usage | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class StreamEnd:
    provider: str
    usage: Usage | None = field(default=None)


LLMChunk = TextDelta | StreamEnd


class LLMError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class LLMClient(Resource, Protocol):
    @property
    def name(self) -> str: ...

    async def complete(self, request: LLMRequest) -> Completion: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]: ...
