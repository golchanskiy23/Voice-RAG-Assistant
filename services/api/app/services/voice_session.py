import asyncio
import base64
import contextlib
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import Any, Literal

import structlog

from app.adapters.stt.base import STT_SAMPLE_RATE, SpeechToText
from app.adapters.vad.base import VoiceActivityDetector
from app.core.errors import AppError
from app.services.audio import AudioFormat, decode_to_pcm
from app.services.voice import AnswerText, AudioChunk, Timeline, VoicePipeline, VoiceSources

log = structlog.get_logger()

Send = Callable[[dict[str, Any]], Awaitable[None]]
InterruptReason = Literal["barge_in", "cancel"]
PRE_ROLL_MS = 300
BYTES_PER_SECOND = STT_SAMPLE_RATE * 2


class VoiceSession:
    def __init__(
        self,
        send: Send,
        stt: SpeechToText,
        vad: VoiceActivityDetector,
        pipeline: VoicePipeline,
        request_id: str | None = None,
        partial_interval: float = 1.0,
        max_utterance_seconds: float = 30.0,
    ) -> None:
        self._send_raw = send
        self._send_lock = asyncio.Lock()
        self._stt = stt
        self._vad = vad
        self._pipeline = pipeline
        self._request_id = request_id
        self._partial_bytes = int(partial_interval * BYTES_PER_SECOND)
        self._max_utterance_bytes = int(max_utterance_seconds * BYTES_PER_SECOND)
        self._format: AudioFormat = "pcm16"
        self._pending = b""
        self._file = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, PRE_ROLL_MS * 32 // vad.frame_bytes))
        self._utterance: bytearray | None = None
        self._partial_at = 0
        self._partial_task: asyncio.Task[None] | None = None
        self._answer_task: asyncio.Task[None] | None = None

    @property
    def answering(self) -> bool:
        return self._answer_task is not None and not self._answer_task.done()

    async def start(self, audio_format: AudioFormat) -> None:
        self._format = audio_format
        self._pending = b""
        self._file.clear()
        self._utterance = None
        self._vad.reset()

    async def feed_audio(self, data: bytes) -> None:
        if self._format != "pcm16":
            self._file.extend(data)
            return
        self._pending += data
        size = self._vad.frame_bytes
        while len(self._pending) >= size:
            frame, self._pending = self._pending[:size], self._pending[size:]
            await self._process_frame(frame)

    async def end(self) -> None:
        if self._format != "pcm16":
            data, self._file = bytes(self._file), bytearray()
            timeline = Timeline()
            try:
                pcm = await decode_to_pcm(data)
            except AppError as exc:
                await self._error(exc.code, exc.message)
                return
            self._respond(pcm=pcm, timeline=timeline)
            return
        if self._utterance is not None:
            await self._finish_utterance()

    async def ask_text(self, question: str) -> None:
        self._respond(question=question, timeline=Timeline())

    async def cancel(self) -> None:
        await self._interrupt("cancel")

    async def close(self) -> None:
        for task in (self._partial_task, self._answer_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _process_frame(self, frame: bytes) -> None:
        event = self._vad.process(frame)
        if event == "speech_start":
            if self.answering:
                await self._interrupt("barge_in")
            self._utterance = bytearray(b"".join(self._pre_roll))
            self._partial_at = 0
        if self._utterance is None:
            self._pre_roll.append(frame)
            return
        self._utterance.extend(frame)
        if event == "speech_end" or len(self._utterance) >= self._max_utterance_bytes:
            await self._finish_utterance()
            return
        self._maybe_partial()

    def _maybe_partial(self) -> None:
        if not self._partial_bytes or self._utterance is None:
            return
        if len(self._utterance) - self._partial_at < self._partial_bytes:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return
        self._partial_at = len(self._utterance)
        self._partial_task = asyncio.create_task(self._partial(bytes(self._utterance)))

    async def _partial(self, pcm: bytes) -> None:
        try:
            transcript = await self._stt.transcribe(pcm)
        except Exception:
            log.warning("stt_partial_failed", stage="stt", exc_info=True)
            return
        if self._utterance is not None and transcript.text:
            await self._send({"type": "partial_transcript", "text": transcript.text})

    async def _finish_utterance(self) -> None:
        timeline = Timeline()
        pcm = bytes(self._utterance or b"")
        self._utterance = None
        self._pre_roll.clear()
        self._vad.reset()
        if self._partial_task is not None:
            self._partial_task.cancel()
        self._respond(pcm=pcm, timeline=timeline)

    def _respond(
        self, *, timeline: Timeline, pcm: bytes | None = None, question: str | None = None
    ) -> None:
        if self._answer_task is not None and not self._answer_task.done():
            self._answer_task.cancel()
        self._answer_task = asyncio.create_task(self._answer(timeline, pcm, question))

    async def _answer(self, timeline: Timeline, pcm: bytes | None, question: str | None) -> None:
        try:
            if pcm is not None:
                transcript = await self._stt.transcribe(pcm)
                timeline.mark("stt")
                await self._send(
                    {
                        "type": "final_transcript",
                        "text": transcript.text,
                        "audio_seconds": round(transcript.audio_seconds, 2),
                    }
                )
                question = transcript.text
            if not question:
                await self._error("empty_transcript", "Речь не распознана")
                return
            await self._speak(question, timeline)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("voice_answer_failed", stage="voice")
            await self._error("voice_error", "Ошибка при подготовке голосового ответа")

    async def _speak(self, question: str, timeline: Timeline) -> None:
        async with aclosing(self._pipeline.answer(question, timeline)) as events:
            async for event in events:
                match event:
                    case AnswerText(text=text):
                        await self._send({"type": "answer_text", "text": text})
                    case AudioChunk(seq=seq, pcm=data, sample_rate=rate):
                        await self._send(
                            {
                                "type": "audio_chunk",
                                "seq": seq,
                                "sample_rate": rate,
                                "format": "pcm16",
                                "data": base64.b64encode(data).decode("ascii"),
                            }
                        )
                        timeline.mark("first_audio")
                    case VoiceSources(sources=sources, confidence=confidence):
                        await self._send(
                            {
                                "type": "sources",
                                "sources": [
                                    {
                                        "filename": s.filename,
                                        "page": s.page,
                                        "section": s.section,
                                        "quote": s.quote,
                                    }
                                    for s in sources
                                ],
                                "confidence": confidence,
                            }
                        )
        timeline.mark("total")
        log.info("voice_answer", stage="voice", latency_ms=timeline.marks)
        await self._send({"type": "done", "latency": timeline.marks})

    async def _interrupt(self, reason: InterruptReason) -> None:
        task = self._answer_task
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await self._send({"type": "interrupted", "reason": reason})

    async def _error(self, code: str, message: str) -> None:
        await self._send(
            {
                "type": "error",
                "error": {"code": code, "message": message, "request_id": self._request_id},
            }
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._send_raw(payload)
