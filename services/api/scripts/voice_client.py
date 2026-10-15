import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import websockets

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.services.audio import decode_to_pcm, pcm_to_wav  # noqa: E402

CHUNK_SECONDS = 0.02
TRAILING_SILENCE_SECONDS = 1.0
STT_RATE = 16_000


async def send_paced(ws: websockets.ClientConnection, audio: bytes, realtime: bool) -> None:
    chunk = int(STT_RATE * 2 * CHUNK_SECONDS)
    for start in range(0, len(audio), chunk):
        await ws.send(audio[start : start + chunk])
        if realtime:
            await asyncio.sleep(CHUNK_SECONDS)


async def stream_pcm(ws: websockets.ClientConnection, pcm: bytes, realtime: bool) -> float:
    await send_paced(ws, pcm, realtime)
    speech_end = time.perf_counter()
    await send_paced(ws, bytes(int(STT_RATE * 2 * TRAILING_SILENCE_SECONDS)), realtime)
    return speech_end


async def converse(
    url: str, *, audio: Path | None, text: str | None, realtime: bool, out: Path | None, quiet: bool
) -> dict[str, object]:
    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        sent_at = time.perf_counter()
        if text is not None:
            await ws.send(json.dumps({"type": "text", "question": text}))
        elif audio is not None:
            pcm = await decode_to_pcm(await asyncio.to_thread(audio.read_bytes))
            await ws.send(json.dumps({"type": "start", "format": "pcm16"}))
            sent_at = await stream_pcm(ws, pcm, realtime)
        chunks: list[bytes] = []
        first_audio_client: float | None = None
        result: dict[str, object] = {"ready": ready}
        async for raw in ws:
            message = json.loads(raw)
            kind = message["type"]
            if kind == "audio_chunk":
                if first_audio_client is None:
                    first_audio_client = (time.perf_counter() - sent_at) * 1000
                chunks.append(base64.b64decode(message["data"]))
                result["sample_rate"] = message["sample_rate"]
            elif not quiet:
                print(json.dumps(message, ensure_ascii=False))
            if kind in ("final_transcript", "sources"):
                result[kind] = message
            if kind in ("done", "error"):
                result[kind] = message
                break
        result["client_first_audio_ms"] = first_audio_client
        if out and chunks:
            wav = pcm_to_wav(b"".join(chunks), int(str(result["sample_rate"])))
            await asyncio.to_thread(out.write_bytes, wav)
            if not quiet:
                print(f"ответ сохранён: {out}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI-клиент /ws/voice")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="wav/ogg/mp3 — любой формат ffmpeg")
    source.add_argument("--text", help="вопрос текстом без STT")
    parser.add_argument("--url", default="ws://localhost:8000/ws/voice")
    parser.add_argument("--out", type=Path, default=Path("answer.wav"))
    parser.add_argument(
        "--fast", action="store_true", help="слать аудио без пауз реального времени"
    )
    args = parser.parse_args()
    result = asyncio.run(
        converse(
            args.url,
            audio=args.audio,
            text=args.text,
            realtime=not args.fast,
            out=args.out,
            quiet=False,
        )
    )
    print(f"первый звук у клиента: {result['client_first_audio_ms']} мс")


if __name__ == "__main__":
    main()
