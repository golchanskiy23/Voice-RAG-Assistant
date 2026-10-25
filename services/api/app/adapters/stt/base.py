from dataclasses import dataclass
from typing import Protocol

from app.adapters.base import Resource

STT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Transcript:
    text: str
    audio_seconds: float


class SpeechToText(Resource, Protocol):
    @property
    def name(self) -> str: ...

    async def transcribe(self, pcm: bytes) -> Transcript: ...
