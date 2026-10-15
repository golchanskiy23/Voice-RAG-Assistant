import asyncio
import math
from array import array

MS_PER_CHAR = 60


class StubTTS:
    def __init__(
        self, sample_rate: int = 24_000, realtime_factor: float = 0.0, frequency: float = 220.0
    ) -> None:
        self._sample_rate = sample_rate
        self._realtime_factor = realtime_factor
        self._frequency = frequency
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> bytes:
        seconds = len(text) * MS_PER_CHAR / 1000
        if self._realtime_factor:
            await asyncio.sleep(seconds * self._realtime_factor)
        count = int(seconds * self._sample_rate)
        step = 2 * math.pi * self._frequency / self._sample_rate
        samples = array("h", (int(3000 * math.sin(i * step)) for i in range(count)))
        return samples.tobytes()

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
