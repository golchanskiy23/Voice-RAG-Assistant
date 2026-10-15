import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.adapters.tts.base import TextToSpeech
from app.services.rag import FinalEvent, RAGService, RetrievedEvent, Source, TokenEvent
from app.services.sentences import SentenceBuffer
from app.services.tts_text import normalize_for_tts


@dataclass
class Timeline:
    start: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        if stage not in self.marks:
            self.marks[stage] = round((time.perf_counter() - self.start) * 1000, 2)


@dataclass(frozen=True)
class AnswerText:
    text: str


@dataclass(frozen=True)
class AudioChunk:
    seq: int
    pcm: bytes
    sample_rate: int


@dataclass(frozen=True)
class VoiceSources:
    sources: list[Source]
    confidence: float


VoiceEvent = AnswerText | AudioChunk | VoiceSources

END_OF_TEXT = None


class VoicePipeline:
    def __init__(
        self,
        rag: RAGService,
        tts: TextToSpeech,
        streaming: bool = True,
        first_chunk_chars: int | None = None,
    ) -> None:
        self._rag = rag
        self._tts = tts
        self._streaming = streaming
        self._first_chunk_chars = first_chunk_chars

    async def answer(self, question: str, timeline: Timeline) -> AsyncGenerator[VoiceEvent]:
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()
        async with asyncio.TaskGroup() as tg:
            producer = tg.create_task(self._produce(question, timeline, sentences))
            tg.create_task(self._synthesize(timeline, sentences, events))
            while (event := await events.get()) is not END_OF_TEXT:
                yield event
            final = await producer
        if final is not None:
            yield VoiceSources(final.sources, final.confidence)

    async def _produce(
        self, question: str, timeline: Timeline, sentences: asyncio.Queue[str | None]
    ) -> FinalEvent | None:
        buffer = SentenceBuffer(first_max_chars=self._first_chunk_chars)
        final: FinalEvent | None = None
        pending: list[str] = []
        async for event in self._rag.stream(question):
            match event:
                case RetrievedEvent():
                    timeline.mark("retrieve")
                case TokenEvent(text=text):
                    timeline.mark("first_token")
                    ready = buffer.feed(text)
                    if self._streaming:
                        for sentence in ready:
                            timeline.mark("first_sentence")
                            await sentences.put(sentence)
                    else:
                        pending.extend(ready)
                case FinalEvent():
                    final = event
        for sentence in [*pending, *buffer.flush()]:
            timeline.mark("first_sentence")
            await sentences.put(sentence)
        await sentences.put(END_OF_TEXT)
        return final

    async def _synthesize(
        self,
        timeline: Timeline,
        sentences: asyncio.Queue[str | None],
        events: asyncio.Queue[VoiceEvent | None],
    ) -> None:
        seq = 0
        while (sentence := await sentences.get()) is not END_OF_TEXT:
            await events.put(AnswerText(sentence))
            spoken = normalize_for_tts(sentence)
            if not spoken:
                continue
            pcm = await self._tts.synthesize(spoken)
            timeline.mark("first_synthesized")
            await events.put(AudioChunk(seq, pcm, self._tts.sample_rate))
            seq += 1
        await events.put(END_OF_TEXT)
