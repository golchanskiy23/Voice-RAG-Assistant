import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.adapters.llm.base import LLMClient


@dataclass(frozen=True)
class Source:
    filename: str
    page: int | None
    section: str | None
    quote: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class FinalEvent:
    sources: list[Source]
    confidence: float
    latency_ms: dict[str, float]


StreamEvent = TokenEvent | FinalEvent


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RAGService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        text = await self._llm.complete(question)
        return Answer(text=text, latency_ms={"llm": elapsed_ms(start)})

    async def stream(self, question: str) -> AsyncGenerator[StreamEvent]:
        start = time.perf_counter()
        latency: dict[str, float] = {}
        async for token in self._llm.stream(question):
            latency.setdefault("first_token", elapsed_ms(start))
            yield TokenEvent(token)
        latency["llm"] = elapsed_ms(start)
        yield FinalEvent(sources=[], confidence=0.0, latency_ms=latency)
