import asyncio
import contextlib
import html
from collections.abc import AsyncIterator
from typing import Literal, Protocol

import structlog

from app.adapters.dialog.base import DialogStore
from app.adapters.stt.base import SpeechToText
from app.adapters.tts.base import TextToSpeech
from app.core.errors import AppError
from app.services.audio import decode_to_pcm, encode_ogg_opus
from app.services.dialog import DialogService
from app.services.rag import Answer
from app.services.tts_text import normalize_for_tts

log = structlog.get_logger()

ChatAction = Literal["typing", "record_voice", "upload_voice"]
ACTION_REFRESH_SECONDS = 4.0
RATE_WINDOW_SECONDS = 60
GREETING = (
    "Отвечаю на вопросы по загруженным документам — текстом или голосом, со ссылкой на источник.\n"
    "Можно уточнять: «а сколько это стоит?». /reset — забыть историю диалога."
)


class Messenger(Protocol):
    async def send_text(self, chat_id: int, text: str) -> None: ...

    async def send_voice(self, chat_id: int, ogg: bytes) -> None: ...

    async def send_action(self, chat_id: int, action: ChatAction) -> None: ...

    async def download(self, file_id: str) -> bytes: ...


def format_answer(answer: Answer, heard: str | None = None) -> str:
    parts = []
    if heard:
        parts.append(f"🎤 <i>{html.escape(heard)}</i>\n")
    parts.append(html.escape(answer.text))
    if answer.sources:
        lines = []
        for number, source in enumerate(answer.sources, start=1):
            where = source.filename
            if source.page is not None:
                where += f", стр. {source.page}"
            if source.section:
                where += f" · {source.section}"
            lines.append(f"{number}. <b>{html.escape(where)}</b> — «{html.escape(source.quote)}»")
        parts.append("\n<b>Источники:</b>\n" + "\n".join(lines))
    return "\n".join(parts)


class TelegramBot:
    def __init__(
        self,
        messenger: Messenger,
        dialog: DialogService,
        store: DialogStore,
        stt: SpeechToText,
        tts: TextToSpeech,
        whitelist: frozenset[int] = frozenset(),
        rate_limit: int = 10,
        max_voice_seconds: int = 60,
    ) -> None:
        self._messenger = messenger
        self._dialog = dialog
        self._store = store
        self._stt = stt
        self._tts = tts
        self._whitelist = whitelist
        self._rate_limit = rate_limit
        self._max_voice_seconds = max_voice_seconds

    async def on_start(self, chat_id: int) -> None:
        if self._allowed(chat_id):
            await self._messenger.send_text(chat_id, GREETING)

    async def on_reset(self, chat_id: int) -> None:
        if self._allowed(chat_id):
            await self._dialog.reset(self._session(chat_id))
            await self._messenger.send_text(chat_id, "История диалога очищена.")

    async def on_text(self, chat_id: int, user_id: int, text: str) -> None:
        if not await self._admit(chat_id, user_id):
            return
        async with self._action(chat_id, "typing"):
            result = await self._dialog.answer(self._session(chat_id), text.strip()[:2000])
        await self._messenger.send_text(chat_id, format_answer(result.answer))

    async def on_voice(self, chat_id: int, user_id: int, file_id: str, duration: int) -> None:
        if not await self._admit(chat_id, user_id):
            return
        if duration > self._max_voice_seconds:
            await self._messenger.send_text(
                chat_id, f"Голосовое длиннее {self._max_voice_seconds} с — разбейте вопрос."
            )
            return
        async with self._action(chat_id, "record_voice"):
            try:
                pcm = await decode_to_pcm(await self._messenger.download(file_id))
            except AppError:
                await self._messenger.send_text(chat_id, "Не удалось прочитать голосовое.")
                return
            heard = (await self._stt.transcribe(pcm)).text
            if not heard:
                await self._messenger.send_text(chat_id, "Не расслышал вопрос, повторите.")
                return
            result = await self._dialog.answer(self._session(chat_id), heard)
            await self._messenger.send_text(chat_id, format_answer(result.answer, heard))
            spoken = normalize_for_tts(result.answer.text)
            if spoken:
                ogg = await encode_ogg_opus(
                    await self._tts.synthesize(spoken), self._tts.sample_rate
                )
                await self._messenger.send_voice(chat_id, ogg)

    async def _admit(self, chat_id: int, user_id: int) -> bool:
        if not self._allowed(chat_id):
            log.info("telegram_denied", stage="telegram", chat_id=chat_id)
            return False
        limited = await self._store.hit_rate_limit(
            f"tg:{user_id}", self._rate_limit, RATE_WINDOW_SECONDS
        )
        if limited:
            await self._messenger.send_text(
                chat_id, f"Не больше {self._rate_limit} вопросов в минуту. Подождите немного."
            )
            return False
        return True

    def _allowed(self, chat_id: int) -> bool:
        return not self._whitelist or chat_id in self._whitelist

    @staticmethod
    def _session(chat_id: int) -> str:
        return f"tg:{chat_id}"

    @contextlib.asynccontextmanager
    async def _action(self, chat_id: int, action: ChatAction) -> AsyncIterator[None]:
        async def send() -> None:
            with contextlib.suppress(Exception):
                await self._messenger.send_action(chat_id, action)

        async def refresh() -> None:
            while True:
                await asyncio.sleep(ACTION_REFRESH_SECONDS)
                await send()

        await send()
        task = asyncio.create_task(refresh())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
