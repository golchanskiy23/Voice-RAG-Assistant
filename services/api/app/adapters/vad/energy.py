import math
from array import array

from app.adapters.vad.base import VadEvent

SAMPLE_RATE = 16_000


def frame_dbfs(frame: bytes) -> float:
    samples = array("h")
    samples.frombytes(frame)
    if not samples:
        return -120.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    return 20 * math.log10(max(rms, 1.0) / 32768.0)


class EnergyVAD:
    def __init__(
        self,
        threshold_dbfs: float = -40.0,
        frame_ms: int = 30,
        min_speech_ms: int = 150,
        min_silence_ms: int = 600,
    ) -> None:
        self._threshold = threshold_dbfs
        self._frame_ms = frame_ms
        self._speech_frames_needed = max(1, min_speech_ms // frame_ms)
        self._silence_frames_needed = max(1, min_silence_ms // frame_ms)
        self.reset()

    @property
    def frame_bytes(self) -> int:
        return SAMPLE_RATE * self._frame_ms // 1000 * 2

    def reset(self) -> None:
        self._speaking = False
        self._voiced_run = 0
        self._silent_run = 0

    def process(self, frame: bytes) -> VadEvent | None:
        voiced = frame_dbfs(frame) > self._threshold
        if not self._speaking:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self._speech_frames_needed:
                self._speaking = True
                self._silent_run = 0
                return "speech_start"
            return None
        self._silent_run = 0 if voiced else self._silent_run + 1
        if self._silent_run >= self._silence_frames_needed:
            self._speaking = False
            self._voiced_run = 0
            return "speech_end"
        return None
