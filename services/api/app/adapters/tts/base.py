from typing import Protocol

from app.adapters.base import Resource


class TextToSpeech(Resource, Protocol):
    @property
    def name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    async def synthesize(self, text: str) -> bytes: ...
