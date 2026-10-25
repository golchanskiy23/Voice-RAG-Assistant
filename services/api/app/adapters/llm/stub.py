import asyncio
import re
from collections.abc import AsyncIterator

from app.services.llm_answer import NO_ANSWER, LLMAnswer


class StubLLMClient:
    def __init__(self, token_delay: float = 0.0) -> None:
        self._token_delay = token_delay
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    async def complete(self, prompt: str) -> str:
        return LLMAnswer(answer=NO_ANSWER, confidence=0.0).model_dump_json()

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for token in re.findall(r"\S+\s*", NO_ANSWER):
            await asyncio.sleep(self._token_delay)
            yield token

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
