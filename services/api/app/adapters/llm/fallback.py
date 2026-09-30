from collections.abc import AsyncIterator

import structlog

from app.adapters.llm.base import Completion, LLMChunk, LLMClient, LLMError, LLMRequest, TextDelta

log = structlog.get_logger()


class FallbackLLMClient:
    def __init__(self, primary: LLMClient, secondary: LLMClient) -> None:
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        return f"{self._primary.name}>{self._secondary.name}"

    async def complete(self, request: LLMRequest) -> Completion:
        try:
            return await self._primary.complete(request)
        except LLMError as exc:
            log.warning("llm_fallback", stage="complete", failed=exc.provider, error=str(exc))
        return await self._secondary.complete(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        started = False
        try:
            async for chunk in self._primary.stream(request):
                started = started or isinstance(chunk, TextDelta)
                yield chunk
            return
        except LLMError as exc:
            if started:
                raise
            log.warning("llm_fallback", stage="stream", failed=exc.provider, error=str(exc))
        async for chunk in self._secondary.stream(request):
            yield chunk

    async def is_ready(self) -> bool:
        return await self._primary.is_ready() or await self._secondary.is_ready()

    async def close(self) -> None:
        await self._primary.close()
        await self._secondary.close()
