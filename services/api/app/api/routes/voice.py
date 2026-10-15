import json
from typing import Any, get_args

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.adapters.stt.base import STT_SAMPLE_RATE, SpeechToText
from app.adapters.tts.base import TextToSpeech
from app.api.deps import rag_service_from_state
from app.services.audio import AudioFormat
from app.services.voice import VoicePipeline
from app.services.voice_session import VoiceSession

router = APIRouter(tags=["voice"])
FORMATS = set(get_args(AudioFormat))


@router.websocket("/ws/voice")
async def voice(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    stt: SpeechToText = state.stt
    tts: TextToSpeech = state.tts
    request_id = websocket.headers.get("x-request-id")

    async def send(payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    session = VoiceSession(
        send,
        stt,
        state.vad_factory(),
        VoicePipeline(
            rag_service_from_state(state),
            tts,
            state.settings.voice_streaming,
            state.settings.voice_first_chunk_chars or None,
        ),
        request_id=request_id,
        partial_interval=state.settings.stt_partial_interval,
        max_utterance_seconds=state.settings.max_utterance_seconds,
    )
    await send(
        {
            "type": "ready",
            "sample_rate_in": STT_SAMPLE_RATE,
            "sample_rate_out": tts.sample_rate,
            "stt": stt.name,
            "tts": tts.name,
        }
    )
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.feed_audio(data)
            elif (text := message.get("text")) is not None:
                await handle_control(session, send, text, request_id)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


async def handle_control(
    session: VoiceSession, send: Any, text: str, request_id: str | None
) -> None:
    try:
        command = json.loads(text)
        kind = command["type"]
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        kind = None
        command = {}
    match kind:
        case "start" if command.get("format", "pcm16") in FORMATS:
            await session.start(command.get("format", "pcm16"))
        case "end":
            await session.end()
        case "text" if isinstance(command.get("question"), str) and command["question"].strip():
            await session.ask_text(command["question"].strip()[:2000])
        case "cancel":
            await session.cancel()
        case _:
            await send(
                {
                    "type": "error",
                    "error": {
                        "code": "bad_message",
                        "message": "Неизвестная команда или неверные поля",
                        "request_id": request_id,
                    },
                }
            )
