import time
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


class RAGService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        text = await self._llm.complete(question)
        llm_ms = (time.perf_counter() - start) * 1000
        return Answer(text=text, latency_ms={"llm": round(llm_ms, 2)})
