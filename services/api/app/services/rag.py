import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.adapters.llm.base import LLMClient
from app.services.llm_answer import NO_ANSWER, AnswerValidationError, LLMAnswer, parse_llm_answer


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


def retry_prompt(prompt: str, previous: str, details: str) -> str:
    return (
        f"{prompt}\n\nПредыдущий ответ не прошёл проверку схемы.\n"
        f"Ответ: {previous}\nОшибки: {details}\nВерни исправленный JSON."
    )


class RAGService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        result = await self._generate(question, chunk_count=0)
        return Answer(
            text=result.answer,
            confidence=result.confidence,
            latency_ms={"llm": elapsed_ms(start)},
        )

    async def _generate(self, prompt: str, chunk_count: int) -> LLMAnswer:
        raw = await self._llm.complete(prompt)
        try:
            return parse_llm_answer(raw, chunk_count)
        except AnswerValidationError as first_error:
            retry_raw = await self._llm.complete(retry_prompt(prompt, raw, first_error.details))
        try:
            return parse_llm_answer(retry_raw, chunk_count)
        except AnswerValidationError:
            return LLMAnswer(answer=NO_ANSWER, confidence=0.0)

    async def stream(self, question: str) -> AsyncGenerator[StreamEvent]:
        start = time.perf_counter()
        latency: dict[str, float] = {}
        async for token in self._llm.stream(question):
            latency.setdefault("first_token", elapsed_ms(start))
            yield TokenEvent(token)
        latency["llm"] = elapsed_ms(start)
        yield FinalEvent(sources=[], confidence=0.0, latency_ms=latency)
