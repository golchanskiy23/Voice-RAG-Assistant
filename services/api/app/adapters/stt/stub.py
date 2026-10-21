from app.adapters.stt.base import STT_SAMPLE_RATE, Transcript


def pcm_seconds(pcm: bytes, sample_rate: int = STT_SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate


class StubSTT:
    def __init__(self, text: str = "Какой максимальный размер файла?") -> None:
        self._text = text
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    async def transcribe(self, pcm: bytes) -> Transcript:
        return Transcript(text=self._text if pcm else "", audio_seconds=pcm_seconds(pcm))

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
