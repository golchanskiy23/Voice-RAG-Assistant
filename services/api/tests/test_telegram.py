import asyncio
import math
import shutil
import subprocess
from array import array
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import TelegramMethod
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.adapters.dialog.memory import InMemoryDialogStore
from app.adapters.stt.stub import StubSTT
from app.adapters.telegram.aiogram import AiogramGateway
from app.adapters.tts.stub import StubTTS
from app.api.routes.telegram import SECRET_HEADER
from app.core.config import PROMPTS_DIR
from app.services.dialog import DialogService, QueryRewriter, load_rewrite_prompt
from app.services.telegram import ChatAction, TelegramBot, format_answer
from tests.factories import build_app, make_settings
from tests.llm_fakes import ScriptedLLM, hit, make_service

ANSWER = '{"answer": "Лимит — 20 МБ <для всех>.", "sources": [1], "confidence": 0.9}'


class FakeMessenger:
    def __init__(self, voice: bytes = b"") -> None:
        self.texts: list[tuple[int, str]] = []
        self.voices: list[tuple[int, bytes]] = []
        self.actions: list[ChatAction] = []
        self.voice = voice

    async def send_text(self, chat_id: int, text: str) -> None:
        self.texts.append((chat_id, text))

    async def send_voice(self, chat_id: int, ogg: bytes) -> None:
        self.voices.append((chat_id, ogg))

    async def send_action(self, chat_id: int, action: ChatAction) -> None:
        self.actions.append(action)

    async def download(self, file_id: str) -> bytes:
        return self.voice


def make_bot(
    messenger: FakeMessenger, whitelist: frozenset[int] = frozenset(), rate_limit: int = 10
) -> TelegramBot:
    llm = ScriptedLLM([ANSWER] * 20)
    store = InMemoryDialogStore()
    dialog = DialogService(
        store,
        QueryRewriter(llm, load_rewrite_prompt(PROMPTS_DIR)),
        make_service(llm, [hit("Файлы до 20 МБ.", filename="faq.md", page=2)]),
    )
    return TelegramBot(
        messenger,
        dialog,
        store,
        StubSTT("Какой лимит файла?"),
        StubTTS(),
        whitelist,
        rate_limit,
    )


def test_text_answer_has_escaped_html_and_sources() -> None:
    messenger = FakeMessenger()

    asyncio.run(make_bot(messenger).on_text(1, 1, "Какой лимит?"))

    [(chat_id, text)] = messenger.texts
    assert chat_id == 1
    assert "Лимит — 20 МБ &lt;для всех&gt;." in text
    assert "<b>faq.md, стр. 2 · Раздел</b>" in text
    assert messenger.actions[0] == "typing"


def test_whitelist_blocks_unknown_chat() -> None:
    messenger = FakeMessenger()

    asyncio.run(make_bot(messenger, whitelist=frozenset({42})).on_text(7, 7, "q"))

    assert messenger.texts == []


def test_rate_limit_per_user() -> None:
    messenger = FakeMessenger()
    bot = make_bot(messenger, rate_limit=2)

    async def run() -> None:
        for _ in range(3):
            await bot.on_text(1, 5, "Какой лимит?")

    asyncio.run(run())
    assert len(messenger.texts) == 3
    assert "Не больше 2 вопросов в минуту" in messenger.texts[-1][1]


def test_reset_clears_history() -> None:
    messenger = FakeMessenger()
    bot = make_bot(messenger)

    async def run() -> None:
        await bot.on_text(1, 1, "Какой лимит?")
        await bot.on_reset(1)

    asyncio.run(run())
    assert messenger.texts[-1][1] == "История диалога очищена."


def tone_ogg(seconds: float = 1.0) -> bytes:
    step = 2 * math.pi * 300 / 16000
    pcm = array("h", (int(8000 * math.sin(i * step)) for i in range(int(16000 * seconds))))
    command = ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1"]
    command += ["-i", "pipe:0", "-c:a", "libopus", "-f", "ogg", "pipe:1"]
    return subprocess.run(command, input=pcm.tobytes(), capture_output=True, check=True).stdout


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
def test_voice_message_gets_text_and_voice_reply() -> None:
    messenger = FakeMessenger(voice=tone_ogg())

    asyncio.run(make_bot(messenger).on_voice(1, 1, "file-1", duration=1))

    [(_, text)] = messenger.texts
    assert text.startswith("🎤 <i>Какой лимит файла?</i>")
    [(_, ogg)] = messenger.voices
    assert ogg[:4] == b"OggS"
    assert messenger.actions[0] == "record_voice"


def test_too_long_voice_is_rejected() -> None:
    messenger = FakeMessenger()

    asyncio.run(make_bot(messenger).on_voice(1, 1, "f", duration=600))

    assert "длиннее" in messenger.texts[0][1]
    assert messenger.voices == []


def test_format_answer_without_sources() -> None:
    from app.services.rag import Answer

    assert format_answer(Answer(text="Нет ответа.")) == "Нет ответа."


class RecordingSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any:
        self.calls.append(method)
        return True

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,  # noqa: ASYNC109
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes]:
        yield b""

    async def close(self) -> None:
        return None


def update(update_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "T"},
            "text": text,
            **(
                {"entities": [{"type": "bot_command", "offset": 0, "length": len(text)}]}
                if text.startswith("/")
                else {}
            ),
        },
    }


def test_aiogram_dispatches_commands_and_text() -> None:
    session = RecordingSession()
    bot = Bot("42:TEST", session=session)
    messenger = FakeMessenger()
    gateway = AiogramGateway(bot, make_bot(messenger))

    async def run() -> None:
        await gateway.process(update(1, "/start"))
        await gateway.process(update(2, "Какой лимит?"))

    asyncio.run(run())
    assert "Отвечаю на вопросы" in messenger.texts[0][1]
    assert "Лимит" in messenger.texts[1][1]


class RecordingProcessor:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def process(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)


def telegram_client(processor: RecordingProcessor | None) -> TestClient:
    settings = make_settings().model_copy(update={"telegram_secret": SecretStr("s3cret")})
    app = build_app(settings=settings)
    client = TestClient(app)
    client.__enter__()
    app.state.telegram = processor
    return client


def test_webhook_checks_secret_and_deduplicates_updates() -> None:
    processor = RecordingProcessor()
    client = telegram_client(processor)
    try:
        bad = client.post("/telegram/webhook", json=update(1, "q"), headers={SECRET_HEADER: "no"})
        first = client.post(
            "/telegram/webhook", json=update(1, "q"), headers={SECRET_HEADER: "s3cret"}
        )
        again = client.post(
            "/telegram/webhook", json=update(1, "q"), headers={SECRET_HEADER: "s3cret"}
        )
        other = client.post(
            "/telegram/webhook", json=update(2, "q"), headers={SECRET_HEADER: "s3cret"}
        )
    finally:
        client.__exit__(None, None, None)

    assert bad.status_code == 401 and bad.json()["error"]["code"] == "unauthorized"
    assert first.status_code == again.status_code == other.status_code == 200
    assert [p["update_id"] for p in processor.payloads] == [1, 2]


def test_webhook_without_bot_is_404() -> None:
    with TestClient(build_app()) as client:
        response = client.post("/telegram/webhook", json=update(1, "q"))

    assert response.status_code == 404


def test_ask_with_session_keeps_history() -> None:
    with TestClient(build_app()) as client:
        for question in ("файлы больше", "а если меньше?"):
            response = client.post("/ask", json={"question": question, "session_id": "abc"})
            assert response.status_code == 200
        store = client.app.state.dialog_store  # type: ignore[attr-defined]
        history = asyncio.run(store.history("api:abc"))

    assert [t.text for t in history][::2] == ["файлы больше", "а если меньше?"]


def test_settings_require_secret_with_token() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_SECRET"):
        make_settings().model_validate(
            {**make_settings().model_dump(), "telegram_token": "1:x", "telegram_secret": None}
        )
