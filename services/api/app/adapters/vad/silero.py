import torch

from app.adapters.vad.base import VadEvent

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512


class SileroVAD:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 600) -> None:
        model, _ = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._model = model
        self._threshold = threshold
        self._silence_windows = max(1, min_silence_ms * SAMPLE_RATE // 1000 // WINDOW_SAMPLES)
        self.reset()

    @property
    def frame_bytes(self) -> int:
        return WINDOW_SAMPLES * 2

    def reset(self) -> None:
        self._model.reset_states()
        self._speaking = False
        self._silent_run = 0

    def process(self, frame: bytes) -> VadEvent | None:
        audio = torch.frombuffer(bytearray(frame), dtype=torch.int16).float() / 32768.0
        with torch.inference_mode():
            probability = float(self._model(audio, SAMPLE_RATE).item())
        voiced = probability >= self._threshold
        if not self._speaking:
            if voiced:
                self._speaking = True
                self._silent_run = 0
                return "speech_start"
            return None
        self._silent_run = 0 if voiced else self._silent_run + 1
        if self._silent_run >= self._silence_windows:
            self._speaking = False
            return "speech_end"
        return None
