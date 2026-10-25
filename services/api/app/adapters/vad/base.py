from typing import Literal, Protocol

VadEvent = Literal["speech_start", "speech_end"]


class VoiceActivityDetector(Protocol):
    @property
    def frame_bytes(self) -> int: ...

    def process(self, frame: bytes) -> VadEvent | None: ...

    def reset(self) -> None: ...
