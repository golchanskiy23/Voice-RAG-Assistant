import time
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict, cast

import structlog

from app.adapters.llm.base import (
    ChatMessage,
    LLMClient,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.adapters.vector_stores.base import SearchHit
from app.services.context import fit_to_budget, make_quote
from app.services.llm_answer import (
    ANSWER_SCHEMA,
    NO_ANSWER,
    AnswerFieldStream,
    AnswerValidationError,
    LLMAnswer,
    parse_llm_answer,
)
from app.services.prompts import AnswerPrompt
from app.services.search import SearchService

log = structlog.get_logger()


class Latency(TypedDict, total=False):
    embed: float
    search: float
    bm25: float
    rerank: float
    first_token: float
    llm: float
    total: float


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
    latency_ms: Latency = field(default_factory=Latency)


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class FinalEvent:
    sources: list[Source]
    confidence: float
    latency_ms: Latency


StreamEvent = TokenEvent | FinalEvent


@dataclass(frozen=True)
class RAGSettings:
    top_k: int = 5
    min_score: float = 0.0
    context_tokens: int = 3000
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass
class Generation:
    result: LLMAnswer
    provider: str | None = None
    usage: list[Usage] = field(default_factory=list)
    attempts: int = 0


@dataclass(frozen=True)
class Retrieval:
    hits: list[SearchHit]
    latency_ms: Latency


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def refusal() -> LLMAnswer:
    return LLMAnswer(answer=NO_ANSWER, confidence=0.0)


def grounded(result: LLMAnswer) -> LLMAnswer:
    return result if result.sources else refusal()


def cited_sources(result: LLMAnswer, hits: Sequence[SearchHit]) -> list[Source]:
    sources: list[Source] = []
    for number in dict.fromkeys(result.sources):
        hit = hits[number - 1]
        sources.append(Source(hit.filename, hit.page, hit.section, make_quote(hit.text)))
    return sources


class RAGService:
    def __init__(
        self,
        search: SearchService,
        llm: LLMClient,
        prompt: AnswerPrompt,
        count_tokens: Callable[[str], int],
        settings: RAGSettings | None = None,
    ) -> None:
        self._search = search
        self._llm = llm
        self._prompt = prompt
        self._count_tokens = count_tokens
        self._settings = settings or RAGSettings()

    async def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        retrieval = await self._retrieve(question)
        latency = retrieval.latency_ms
        generation = Generation(refusal())
        if retrieval.hits:
            llm_start = time.perf_counter()
            generation = await self._generate(self._messages(question, retrieval.hits), retrieval)
            latency["llm"] = elapsed_ms(llm_start)
        result = grounded(generation.result)
        latency["total"] = elapsed_ms(start)
        self._log("rag_answer", question, retrieval, generation, result, latency)
        return Answer(
            text=result.answer,
            sources=cited_sources(result, retrieval.hits),
            confidence=result.confidence,
            latency_ms=latency,
        )

    async def stream(self, question: str) -> AsyncGenerator[StreamEvent]:
        start = time.perf_counter()
        retrieval = await self._retrieve(question)
        latency = retrieval.latency_ms
        if not retrieval.hits:
            latency["total"] = elapsed_ms(start)
            self._log("rag_stream", question, retrieval, Generation(refusal()), refusal(), latency)
            yield TokenEvent(NO_ANSWER)
            yield FinalEvent(sources=[], confidence=0.0, latency_ms=latency)
            return

        llm_start = time.perf_counter()
        answer_field = AnswerFieldStream()
        generation = Generation(refusal(), attempts=1)
        async for chunk in self._llm.stream(
            self._request(self._messages(question, retrieval.hits))
        ):
            match chunk:
                case TextDelta(text=text):
                    if decoded := answer_field.feed(text):
                        latency.setdefault("first_token", elapsed_ms(llm_start))
                        yield TokenEvent(decoded)
                case StreamEnd(provider=provider, usage=usage):
                    generation.provider = provider
                    generation.usage = [usage] if usage else []
        latency["llm"] = elapsed_ms(llm_start)
        try:
            generation.result = grounded(parse_llm_answer(answer_field.raw, len(retrieval.hits)))
        except AnswerValidationError as exc:
            log.warning("llm_answer_invalid", stage="stream", errors=exc.details)
        latency["total"] = elapsed_ms(start)
        self._log("rag_stream", question, retrieval, generation, generation.result, latency)
        yield FinalEvent(
            sources=cited_sources(generation.result, retrieval.hits),
            confidence=generation.result.confidence,
            latency_ms=latency,
        )

    async def _retrieve(self, question: str) -> Retrieval:
        found = await self._search.search(question, self._settings.top_k)
        relevant = [hit for hit in found.hits if hit.score >= self._settings.min_score]
        budget = self._settings.context_tokens
        hits = fit_to_budget(relevant, budget, self._count_tokens, self._prompt.render_chunk)
        latency = cast(Latency, dict(found.latency_ms))
        return Retrieval(hits=hits, latency_ms=latency)

    def _messages(self, question: str, hits: Sequence[SearchHit]) -> tuple[ChatMessage, ...]:
        return self._prompt.messages(question, hits)

    def _request(self, messages: tuple[ChatMessage, ...]) -> LLMRequest:
        return LLMRequest(
            messages=messages,
            json_schema=ANSWER_SCHEMA,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
        )

    async def _generate(
        self, messages: tuple[ChatMessage, ...], retrieval: Retrieval
    ) -> Generation:
        generation = Generation(refusal())
        chunk_count = len(retrieval.hits)
        for attempt in (1, 2):
            completion = await self._llm.complete(self._request(messages))
            generation.attempts = attempt
            generation.provider = completion.provider
            if completion.usage:
                generation.usage.append(completion.usage)
            try:
                generation.result = parse_llm_answer(completion.text, chunk_count)
                return generation
            except AnswerValidationError as exc:
                log.warning(
                    "llm_answer_invalid", stage="complete", attempt=attempt, errors=exc.details
                )
                messages = self._prompt.retry_messages(messages, completion.text, exc.details)
        return generation

    def _log(
        self,
        event: str,
        question: str,
        retrieval: Retrieval,
        generation: Generation,
        result: LLMAnswer,
        latency: Latency,
    ) -> None:
        log.debug("rag_question", question=question)
        log.info(
            event,
            stage="answer",
            chunk_ids=[str(hit.chunk_id) for hit in retrieval.hits],
            sources=result.sources,
            answered=result.answer != NO_ANSWER,
            provider=generation.provider,
            prompt_version=self._prompt.version,
            attempts=generation.attempts,
            prompt_tokens=sum(u.prompt_tokens for u in generation.usage),
            completion_tokens=sum(u.completion_tokens for u in generation.usage),
            latency_ms=latency,
        )
