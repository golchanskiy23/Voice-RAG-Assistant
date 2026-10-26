import asyncio

from faster_whisper import WhisperModel

from app.adapters.stt.base import STT_SAMPLE_RATE, Transcript
from app.adapters.stt.stub import pcm_seconds

PCM16_SCALE = 32768.0


class FasterWhisperSTT:
    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 1,
        cpu_threads: int = 0,
    ) -> None:
        self._model = WhisperModel(
            model, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )
        self._language = language
        self._beam_size = beam_size
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return "faster-whisper"

    async def transcribe(self, pcm: bytes) -> Transcript:
        if not pcm:
            return Transcript("", 0.0)
        async with self._lock:
            text = await asyncio.to_thread(self._transcribe, pcm)
        return Transcript(text=text, audio_seconds=pcm_seconds(pcm, STT_SAMPLE_RATE))

    def _transcribe(self, pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / PCM16_SCALE
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def warm_up(self) -> None:
        await self.transcribe(bytes(STT_SAMPLE_RATE * 2))

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
