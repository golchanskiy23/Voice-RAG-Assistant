import asyncio
import re
from collections.abc import AsyncIterator

NO_ANSWER = "В документах нет ответа на этот вопрос."


class StubLLMClient:
    def __init__(self, token_delay: float = 0.0) -> None:
        self._token_delay = token_delay
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    async def complete(self, prompt: str) -> str:
        return NO_ANSWER

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for token in re.findall(r"\S+\s*", NO_ANSWER):
            await asyncio.sleep(self._token_delay)
            yield token

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
