from collections.abc import AsyncIterator
from typing import Protocol

from app.adapters.base import Resource


class LLMClient(Resource, Protocol):
    @property
    def name(self) -> str: ...

    async def complete(self, prompt: str) -> str: ...

    def stream(self, prompt: str) -> AsyncIterator[str]: ...
