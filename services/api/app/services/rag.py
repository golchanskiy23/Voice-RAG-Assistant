import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict, cast

import structlog

from app.adapters.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.adapters.vector_stores.base import SearchHit
from app.repositories.document_texts import DocumentTexts
from app.services.context import fit_to_budget, make_quote
from app.services.llm_answer import (
    ANSWER_SCHEMA,
    NO_ANSWER,
    AnswerFieldStream,
    AnswerValidationError,
    LLMAnswer,
    parse_llm_answer,
)
from app.services.prompts import AnswerPrompt, GeneralPrompt
from app.services.search import SearchService

log = structlog.get_logger()
CHARS_PER_TOKEN = 3
LOGGED_CHUNKS = 20
GENERAL_PREFIX = "📚 В документах этого нет. Из общих знаний модели:\n"


class Latency(TypedDict, total=False):
    documents: float
    general: float
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


@dataclass(frozen=True)
class RetrievedEvent:
    chunks: int


StreamEvent = RetrievedEvent | TokenEvent | FinalEvent


@dataclass(frozen=True)
class RAGSettings:
    top_k: int = 5
    min_score: float = 0.0
    context_tokens: int = 3000
    full_context_tokens: int = 0
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
        texts: DocumentTexts | None = None,
        general: GeneralPrompt | None = None,
    ) -> None:
        self._texts = texts
        self._general = general
        self._search = search
        self._llm = llm
        self._prompt = prompt
        self._count_tokens = count_tokens
        self._settings = settings or RAGSettings()

    async def answer(
        self, question: str, document_ids: Sequence[uuid.UUID] | None = None
    ) -> Answer:
        start = time.perf_counter()
        retrieval = await self._retrieve(question, document_ids)
        latency = retrieval.latency_ms
        generation = Generation(refusal())
        if retrieval.hits:
            llm_start = time.perf_counter()
            generation = await self._generate(self._messages(question, retrieval.hits), retrieval)
            latency["llm"] = elapsed_ms(llm_start)
        result = grounded(generation.result)
        text = result.answer
        if self._general is not None and text == NO_ANSWER:
            general_start = time.perf_counter()
            general = await self._general_answer(self._general, question)
            latency["general"] = elapsed_ms(general_start)
            text = f"{GENERAL_PREFIX}{general}" if general else text
        latency["total"] = elapsed_ms(start)
        self._log("rag_answer", question, retrieval, generation, result, latency)
        return Answer(
            text=text,
            sources=cited_sources(result, retrieval.hits),
            confidence=result.confidence,
            latency_ms=latency,
        )

    async def stream(self, question: str) -> AsyncGenerator[StreamEvent]:
        start = time.perf_counter()
        retrieval = await self._retrieve(question)
        latency = retrieval.latency_ms
        yield RetrievedEvent(len(retrieval.hits))
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

    async def _retrieve(
        self, question: str, document_ids: Sequence[uuid.UUID] | None = None
    ) -> Retrieval:
        if self._texts is not None and self._settings.full_context_tokens > 0:
            started = time.perf_counter()
            whole = await self._whole_documents(self._texts, question, document_ids)
            if whole:
                return Retrieval(hits=whole, latency_ms={"documents": elapsed_ms(started)})
        found = await self._search.search(question, self._settings.top_k, document_ids)
        relevant = [hit for hit in found.hits if hit.score >= self._settings.min_score]
        budget = self._settings.context_tokens
        hits = fit_to_budget(relevant, budget, self._count_tokens, self._prompt.render_chunk)
        latency = cast(Latency, dict(found.latency_ms))
        return Retrieval(hits=hits, latency_ms=latency)

    async def _general_answer(self, prompt: GeneralPrompt, question: str) -> str:
        request = LLMRequest(
            messages=prompt.messages(question),
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
        )
        try:
            completion = await self._llm.complete(request)
        except LLMError:
            log.warning("general_answer_failed", stage="general")
            return ""
        log.info(
            "general_answer",
            stage="general",
            provider=completion.provider,
            prompt_version=prompt.version,
        )
        return completion.text.strip()

    async def _whole_documents(
        self,
        texts: DocumentTexts,
        question: str,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> list[SearchHit]:
        sizes = await texts.sizes()
        budget = self._settings.full_context_tokens * CHARS_PER_TOKEN
        candidates = [d for d in document_ids if d in sizes] if document_ids else list(sizes)
        if sum(sizes[d] for d in candidates) > budget:
            if document_ids:
                return []
            found = await self._search.search(question, self._settings.top_k * 4)
            candidates = []
            for document_id in dict.fromkeys(hit.document_id for hit in found.hits):
                if sum(sizes[d] for d in candidates) + sizes[document_id] <= budget:
                    candidates.append(document_id)
        return await texts.chunks(candidates) if candidates else []

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
            chunks=len(retrieval.hits),
            documents=len({hit.document_id for hit in retrieval.hits}),
            chunk_ids=[str(hit.chunk_id) for hit in retrieval.hits[:LOGGED_CHUNKS]],
            sources=result.sources,
            answered=result.answer != NO_ANSWER,
            provider=generation.provider,
            prompt_version=self._prompt.version,
            attempts=generation.attempts,
            prompt_tokens=sum(u.prompt_tokens for u in generation.usage),
            completion_tokens=sum(u.completion_tokens for u in generation.usage),
            latency_ms=latency,
        )
