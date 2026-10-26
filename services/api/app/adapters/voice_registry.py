from collections.abc import Callable

from app.adapters.registry import Registry
from app.adapters.stt.base import SpeechToText
from app.adapters.stt.stub import StubSTT
from app.adapters.tts.base import TextToSpeech
from app.adapters.tts.stub import StubTTS
from app.adapters.vad.base import VoiceActivityDetector
from app.adapters.vad.energy import EnergyVAD
from app.core.config import Settings


def faster_whisper(settings: Settings) -> SpeechToText:
    from app.adapters.stt.faster_whisper import FasterWhisperSTT

    return FasterWhisperSTT(
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
    )


def silero_tts(settings: Settings) -> TextToSpeech:
    from app.adapters.tts.silero import SileroTTS

    return SileroTTS(speaker=settings.tts_speaker, sample_rate=settings.tts_sample_rate)


def energy_vad(settings: Settings) -> VoiceActivityDetector:
    return EnergyVAD(
        threshold_dbfs=settings.vad_threshold_dbfs, min_silence_ms=settings.vad_min_silence_ms
    )


def silero_vad(settings: Settings) -> VoiceActivityDetector:
    from app.adapters.vad.silero import SileroVAD

    return SileroVAD(min_silence_ms=settings.vad_min_silence_ms)


stt_registry: Registry[Settings, SpeechToText] = Registry("stt")
stt_registry.register("stub", lambda _: StubSTT())
stt_registry.register("faster-whisper", faster_whisper)

tts_registry: Registry[Settings, TextToSpeech] = Registry("tts")
tts_registry.register(
    "stub",
    lambda s: StubTTS(sample_rate=s.tts_sample_rate, realtime_factor=s.stub_tts_realtime_factor),
)
tts_registry.register("silero", silero_tts)

vad_registry: Registry[Settings, VoiceActivityDetector] = Registry("vad")
vad_registry.register("energy", energy_vad)
vad_registry.register("silero", silero_vad)


def vad_factory(settings: Settings) -> Callable[[], VoiceActivityDetector]:
    return lambda: vad_registry.create(settings.vad, settings)
