import asyncio
import io
import wave
from typing import Literal

from app.core.errors import AppError

AudioFormat = Literal["pcm16", "ogg", "wav"]
STT_RATE = 16_000
FFMPEG_TIMEOUT_SECONDS = 30.0


async def decode_to_pcm(data: bytes, sample_rate: int = STT_RATE) -> bytes:
    return await run_ffmpeg(
        data,
        [
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
        ],
    )


async def encode_ogg_opus(pcm: bytes, sample_rate: int) -> bytes:
    return await run_ffmpeg(
        pcm,
        [
            *("-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"),
            *("-c:a", "libopus", "-b:a", "32k", "-f", "ogg"),
        ],
    )


async def run_ffmpeg(data: bytes, arguments: list[str]) -> bytes:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *arguments,
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(FFMPEG_TIMEOUT_SECONDS):
            output, errors = await process.communicate(data)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise AppError("audio_timeout", "Обработка аудио заняла слишком долго", 422) from None
    if process.returncode != 0:
        message = errors.decode(errors="replace").strip().splitlines()[-1:] or ["unknown"]
        raise AppError("bad_audio", f"Не удалось обработать аудио: {message[0]}", 422)
    return output


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def read_wav_pcm16(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("нужен WAV PCM16 mono")
        return wav.readframes(wav.getnframes()), wav.getframerate()
