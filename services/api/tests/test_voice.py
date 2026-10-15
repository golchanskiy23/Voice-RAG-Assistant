import asyncio
import base64
import json
import math
import shutil
import subprocess
from array import array
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.adapters.llm.base import LLMChunk, LLMRequest, StreamEnd, TextDelta
from app.adapters.stt.base import Transcript
from app.adapters.stt.stub import StubSTT
from app.adapters.tts.stub import StubTTS
from app.adapters.vad.energy import EnergyVAD, frame_dbfs
from app.services.audio import decode_to_pcm, pcm_to_wav
from app.services.voice import AnswerText, AudioChunk, Timeline, VoicePipeline, VoiceSources
from app.services.voice_session import VoiceSession
from tests.factories import build_app
from tests.llm_fakes import ScriptedLLM, hit, make_service
from tests.test_search import FAQ, upload

RATE = 16_000
ANSWER = json.dumps(
    {
        "answer": "Файлы больше 20 МБ отклоняются. Это правило из FAQ.",
        "sources": [1],
        "confidence": 0.9,
    },
    ensure_ascii=False,
)


def tone(seconds: float, amplitude: int = 8000, rate: int = RATE) -> bytes:
    step = 2 * math.pi * 300 / rate
    return array(
        "h", (int(amplitude * math.sin(i * step)) for i in range(int(seconds * rate)))
    ).tobytes()


def silence(seconds: float) -> bytes:
    return bytes(int(seconds * RATE) * 2)


def test_energy_vad_detects_start_and_end() -> None:
    vad = EnergyVAD(min_speech_ms=90, min_silence_ms=300)
    audio = silence(0.3) + tone(0.6) + silence(0.6)
    frames = [audio[i : i + vad.frame_bytes] for i in range(0, len(audio), vad.frame_bytes)]

    events = [(i, e) for i, f in enumerate(frames) if (e := vad.process(f))]

    assert [e for _, e in events] == ["speech_start", "speech_end"]
    start, end = (i * 30 for i, _ in events)
    assert 300 <= start <= 420
    assert 900 + 270 <= end <= 900 + 330
    assert frame_dbfs(silence(0.03)) < -80 < frame_dbfs(tone(0.03))


class SlowLLM(ScriptedLLM):
    def __init__(self, delay: float) -> None:
        super().__init__([ANSWER])
        self.delay = delay
        self.finished_at: float | None = None

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        for i in range(0, len(ANSWER), 4):
            await asyncio.sleep(self.delay)
            yield TextDelta(ANSWER[i : i + 4])
        self.finished_at = asyncio.get_running_loop().time()
        yield StreamEnd("slow")


def run_pipeline(streaming: bool) -> tuple[list[object], Timeline, SlowLLM, list[float]]:
    llm = SlowLLM(delay=0.005)
    pipeline = VoicePipeline(
        make_service(llm, [hit("Файлы больше 20 МБ отклоняются.")]), StubTTS(), streaming
    )
    timeline = Timeline()
    audio_times: list[float] = []

    async def run() -> list[object]:
        events: list[object] = []
        async for event in pipeline.answer("лимит?", timeline):
            if isinstance(event, AudioChunk):
                audio_times.append(asyncio.get_running_loop().time())
            events.append(event)
        return events

    return asyncio.run(run()), timeline, llm, audio_times


def test_first_audio_is_ready_before_llm_finishes() -> None:
    events, timeline, llm, audio_times = run_pipeline(streaming=True)

    texts = [e.text for e in events if isinstance(e, AnswerText)]
    assert texts == ["Файлы больше 20 МБ отклоняются.", "Это правило из FAQ."]
    assert llm.finished_at is not None and audio_times[0] < llm.finished_at
    assert isinstance(events[-1], VoiceSources)
    assert events[-1].sources[0].filename == "guide.md"
    marks = timeline.marks
    assert marks["retrieve"] <= marks["first_token"] <= marks["first_sentence"]
    assert marks["first_sentence"] <= marks["first_synthesized"]


def test_non_streaming_mode_waits_for_whole_answer() -> None:
    _, _, llm, audio_times = run_pipeline(streaming=False)

    assert llm.finished_at is not None and audio_times[0] >= llm.finished_at


def test_audio_is_synthesized_from_normalized_text() -> None:
    spoken: list[str] = []

    class RecordingTTS(StubTTS):
        async def synthesize(self, text: str) -> bytes:
            spoken.append(text)
            return await super().synthesize(text)

    pipeline = VoicePipeline(make_service(ScriptedLLM([ANSWER]), [hit("x")]), RecordingTTS())

    async def run() -> None:
        async for _ in pipeline.answer("q", Timeline()):
            pass

    asyncio.run(run())
    assert spoken[0] == "Файлы больше двадцать мегабайт отклоняются."


class RecordingSTT(StubSTT):
    def __init__(self) -> None:
        super().__init__("Какие файлы отклоняются?")
        self.calls: list[int] = []

    async def transcribe(self, pcm: bytes) -> Transcript:
        self.calls.append(len(pcm))
        return await super().transcribe(pcm)


def make_session(
    llm: ScriptedLLM, stt: StubSTT | None = None
) -> tuple[VoiceSession, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    pipeline = VoicePipeline(make_service(llm, [hit("Файлы больше 20 МБ отклоняются.")]), StubTTS())
    session = VoiceSession(
        send,
        stt or RecordingSTT(),
        EnergyVAD(min_speech_ms=60, min_silence_ms=300),
        pipeline,
        partial_interval=0.5,
    )
    return session, sent


async def feed(session: VoiceSession, audio: bytes, chunk: int = 3200) -> None:
    for i in range(0, len(audio), chunk):
        await session.feed_audio(audio[i : i + chunk])
        await asyncio.sleep(0)


async def wait_for(sent: list[dict[str, Any]], kind: str) -> None:
    async with asyncio.timeout(5):
        while not any(m["type"] == kind for m in sent):  # noqa: ASYNC110
            await asyncio.sleep(0.01)


def test_session_vad_partial_final_answer_and_done() -> None:
    stt = RecordingSTT()
    session, sent = make_session(ScriptedLLM([ANSWER]), stt)

    async def run() -> None:
        await feed(session, silence(0.3) + tone(1.5) + silence(0.5))
        await wait_for(sent, "done")
        await session.close()

    asyncio.run(run())
    kinds = [m["type"] for m in sent]
    assert "partial_transcript" in kinds
    assert kinds.index("final_transcript") < kinds.index("answer_text") < kinds.index("audio_chunk")
    assert kinds[-2:] == ["sources", "done"]
    final = next(m for m in sent if m["type"] == "final_transcript")
    assert final["text"] == "Какие файлы отклоняются?"
    assert 1.6 <= final["audio_seconds"] <= 2.2
    chunk = next(m for m in sent if m["type"] == "audio_chunk")
    assert chunk["sample_rate"] == 24_000 and base64.b64decode(chunk["data"])
    latency = sent[-1]["latency"]
    assert {"stt", "retrieve", "first_token", "first_sentence", "first_audio", "total"} <= set(
        latency
    )
    assert latency["stt"] <= latency["first_audio"] <= latency["total"]


class EndlessLLM(ScriptedLLM):
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        yield TextDelta('{"answer": "Начинаю длинный ответ. ')
        while True:
            await asyncio.sleep(0.01)
            yield TextDelta("ещё ")


def test_barge_in_cancels_current_answer() -> None:
    session, sent = make_session(EndlessLLM([]))

    async def run() -> None:
        await session.ask_text("вопрос")
        await wait_for(sent, "answer_text")
        assert session.answering
        await feed(session, tone(0.3))
        await wait_for(sent, "interrupted")
        assert not session.answering
        await session.close()

    asyncio.run(run())
    interrupted = next(m for m in sent if m["type"] == "interrupted")
    assert interrupted["reason"] == "barge_in"
    assert "done" not in [m["type"] for m in sent]


def test_text_question_skips_stt() -> None:
    stt = RecordingSTT()
    session, sent = make_session(ScriptedLLM([ANSWER]), stt)

    async def run() -> None:
        await session.ask_text("Какой лимит?")
        await wait_for(sent, "done")

    asyncio.run(run())
    assert stt.calls == []
    assert "final_transcript" not in [m["type"] for m in sent]
    assert "stt" not in sent[-1]["latency"]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")


def ogg_from(pcm: bytes) -> bytes:
    return subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=True,
    ).stdout


@needs_ffmpeg
def test_ffmpeg_decodes_wav_and_ogg_to_16k_pcm() -> None:
    pcm = tone(1.0)

    from_wav = asyncio.run(decode_to_pcm(pcm_to_wav(pcm, RATE)))
    from_ogg = asyncio.run(decode_to_pcm(ogg_from(pcm)))

    assert from_wav == pcm
    assert abs(len(from_ogg) - len(pcm)) < RATE * 2 * 0.1


@needs_ffmpeg
def test_websocket_ogg_round_trip() -> None:
    with TestClient(build_app(stt_factory=RecordingSTT)) as client:
        upload(client, "faq.md", FAQ)
        with client.websocket_connect("/ws/voice") as ws:
            ready = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "start", "format": "ogg"}))
            ws.send_bytes(ogg_from(tone(1.0)))
            ws.send_text(json.dumps({"type": "end"}))
            messages: list[dict[str, Any]] = []
            while not messages or messages[-1]["type"] not in ("done", "error"):
                messages.append(json.loads(ws.receive_text()))

    assert ready["type"] == "ready" and ready["sample_rate_in"] == 16_000
    kinds = [m["type"] for m in messages]
    assert kinds[0] == "final_transcript"
    assert "audio_chunk" in kinds and kinds[-1] == "done"
    sources = next(m for m in messages if m["type"] == "sources")
    assert sources["sources"][0]["filename"] == "faq.md"


def test_websocket_rejects_unknown_command() -> None:
    with TestClient(build_app()) as client, client.websocket_connect("/ws/voice") as ws:
        ws.receive_text()
        ws.send_text("не json")
        error = json.loads(ws.receive_text())

    assert error["type"] == "error"
    assert error["error"]["code"] == "bad_message"
