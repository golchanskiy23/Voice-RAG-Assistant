import asyncio

import torch

SILERO_SAMPLE_RATES = (8_000, 24_000, 48_000)


class SileroTTS:
    def __init__(
        self,
        speaker: str = "xenia",
        sample_rate: int = 24_000,
        model: str = "v4_ru",
        device: str = "cpu",
        threads: int = 2,
    ) -> None:
        if sample_rate not in SILERO_SAMPLE_RATES:
            raise ValueError(f"Silero поддерживает {SILERO_SAMPLE_RATES}, а не {sample_rate}")
        torch.set_num_threads(threads)
        loaded, _ = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-models",
            "silero_tts",
            language="ru",
            speaker=model,
            trust_repo=True,
        )
        self._model = loaded.to(torch.device(device))
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return "silero"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(self, text: str) -> bytes:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize, text)

    def _synthesize(self, text: str) -> bytes:
        with torch.inference_mode():
            audio = self._model.apply_tts(
                text=text, speaker=self._speaker, sample_rate=self._sample_rate
            )
        pcm: bytes = (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
        return pcm

    async def warm_up(self) -> None:
        await self.synthesize("Привет.")

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
