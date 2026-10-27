import asyncio
import contextlib
import html
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePath
from typing import Literal, Protocol

import structlog

from app.adapters.dialog.base import DialogStore
from app.adapters.llm.base import LLMError
from app.adapters.stt.base import SpeechToText
from app.adapters.tts.base import TextToSpeech
from app.core.errors import AppError
from app.repositories.documents import StoredDocument
from app.services.audio import decode_to_pcm, encode_ogg_opus
from app.services.dialog import DialogService
from app.services.ingestion import EXTENSIONS, MAX_UPLOAD_BYTES, UploadResult
from app.services.rag import Answer
from app.services.tts_text import normalize_for_tts

log = structlog.get_logger()

ChatAction = Literal["typing", "record_voice", "upload_voice", "upload_document"]
ACTION_REFRESH_SECONDS = 4.0
RATE_WINDOW_SECONDS = 60
GREETING = (
    "Отвечаю на вопросы по загруженным документам — текстом или голосом, со ссылкой на источник.\n"
    "Можно уточнять: «а сколько это стоит?».\n\n"
    "Пришлите PDF или Markdown файлом — добавлю его в базу; подпись к файлу — вопрос по нему.\n"
    "/docs — список документов, /delete N — удалить документ N из списка, "
    "/reset — забыть историю диалога."
)
LLM_UNAVAILABLE = "Модель сейчас недоступна, попробуйте позже."
UPLOAD_STATUS = {
    "created": "добавлен в базу",
    "replaced": "обновлён — старая версия заменена",
    "exists": "уже есть в базе",
}


class Messenger(Protocol):
    async def send_text(self, chat_id: int, text: str) -> None: ...

    async def send_voice(self, chat_id: int, ogg: bytes) -> None: ...

    async def send_action(self, chat_id: int, action: ChatAction) -> None: ...

    async def download(self, file_id: str) -> bytes: ...


class Library(Protocol):
    async def upload(self, filename: str, content: bytes) -> UploadResult: ...

    async def list_documents(self) -> list[StoredDocument]: ...

    async def delete(self, document_id: uuid.UUID) -> None: ...


def format_upload(result: UploadResult) -> str:
    document = result.document
    details = f"фрагментов: {document.chunks}"
    if document.pages is not None:
        details = f"страниц: {document.pages}, " + details
    return f"📄 <b>{html.escape(document.filename)}</b> {UPLOAD_STATUS[result.status]} ({details})."


def format_documents(documents: list[StoredDocument]) -> str:
    if not documents:
        return "База пуста. Пришлите PDF или Markdown файлом."
    lines = [
        f"{number}. <b>{html.escape(d.filename)}</b> — фрагментов: {d.chunks}"
        for number, d in enumerate(documents, start=1)
    ]
    return "<b>Документы в базе:</b>\n" + "\n".join(lines)


def format_answer(answer: Answer, heard: str | None = None, with_sources: bool = True) -> str:
    parts = []
    if heard:
        parts.append(f"🎤 <i>{html.escape(heard)}</i>\n")
    parts.append(html.escape(answer.text))
    if answer.sources and with_sources:
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
        library: Library | None = None,
        voice_reply: bool = True,
    ) -> None:
        self._messenger = messenger
        self._dialog = dialog
        self._store = store
        self._stt = stt
        self._tts = tts
        self._whitelist = whitelist
        self._rate_limit = rate_limit
        self._max_voice_seconds = max_voice_seconds
        self._library = library
        self._voice_reply = voice_reply

    async def on_start(self, chat_id: int) -> None:
        if self._allowed(chat_id):
            await self._messenger.send_text(chat_id, GREETING)

    async def on_reset(self, chat_id: int) -> None:
        if self._allowed(chat_id):
            await self._dialog.reset(self._session(chat_id))
            await self._messenger.send_text(chat_id, "История диалога очищена.")

    async def on_text(self, chat_id: int, user_id: int, text: str) -> None:
        if await self._admit(chat_id, user_id):
            await self._answer_text(chat_id, text)

    async def on_docs(self, chat_id: int) -> None:
        if not self._allowed(chat_id):
            return
        if self._library is None:
            await self._messenger.send_text(chat_id, "Список документов недоступен.")
            return
        documents = await self._library.list_documents()
        await self._messenger.send_text(chat_id, format_documents(documents))

    async def on_delete(self, chat_id: int, argument: str | None) -> None:
        if not self._allowed(chat_id) or self._library is None or not self._whitelist:
            return
        documents = await self._library.list_documents()
        number = int(argument) if argument and argument.strip().isdigit() else 0
        if not 1 <= number <= len(documents):
            await self._messenger.send_text(
                chat_id, "Укажите номер документа из /docs, например: /delete 2"
            )
            return
        document = documents[number - 1]
        await self._library.delete(document.id)
        await self._messenger.send_text(
            chat_id, f"🗑 <b>{html.escape(document.filename)}</b> удалён из базы."
        )

    async def on_document(
        self,
        chat_id: int,
        user_id: int,
        file_id: str,
        filename: str,
        size: int | None,
        caption: str | None = None,
    ) -> None:
        if not await self._admit(chat_id, user_id):
            return
        if self._library is None or not self._whitelist:
            await self._messenger.send_text(
                chat_id,
                "Загрузка документов доступна только чатам из TELEGRAM_WHITELIST. "
                f"Ваш chat_id: <code>{chat_id}</code>",
            )
            return
        if PurePath(filename).suffix.lower() not in EXTENSIONS:
            await self._messenger.send_text(chat_id, "Принимаю только PDF и Markdown (.pdf, .md).")
            return
        if size is not None and size > MAX_UPLOAD_BYTES:
            limit = MAX_UPLOAD_BYTES // (1024 * 1024)
            await self._messenger.send_text(chat_id, f"Файл больше {limit} МБ.")
            return
        async with self._action(chat_id, "upload_document"):
            try:
                content = await self._messenger.download(file_id)
                result = await self._library.upload(filename, content)
            except AppError as exc:
                await self._messenger.send_text(chat_id, html.escape(exc.message))
                return
            except Exception:
                log.exception("telegram_upload_failed", stage="telegram")
                await self._messenger.send_text(chat_id, "Не удалось добавить документ.")
                return
        log.info(
            "telegram_document_indexed",
            stage="telegram",
            document_id=str(result.document.id),
            status=result.status,
            chunks=result.document.chunks,
        )
        await self._dialog.note_document(self._session(chat_id), result.document.filename)
        await self._messenger.send_text(chat_id, format_upload(result))
        if caption and caption.strip():
            await self._answer_text(chat_id, caption, [result.document.id])

    async def _answer_text(
        self, chat_id: int, text: str, document_ids: list[uuid.UUID] | None = None
    ) -> None:
        async with self._action(chat_id, "typing"):
            try:
                result = await self._dialog.answer(
                    self._session(chat_id), text.strip()[:2000], document_ids
                )
            except LLMError:
                log.exception("telegram_llm_failed", stage="telegram")
                await self._messenger.send_text(chat_id, LLM_UNAVAILABLE)
                return
        await self._messenger.send_text(chat_id, format_answer(result.answer))

    async def on_voice(self, chat_id: int, user_id: int, file_id: str, duration: int) -> None:
        if not await self._admit(chat_id, user_id):
            return
        if duration > self._max_voice_seconds:
            await self._messenger.send_text(
                chat_id, f"Голосовое длиннее {self._max_voice_seconds} с — разбейте вопрос."
            )
            return
        async with self._action(chat_id, "record_voice" if self._voice_reply else "typing"):
            try:
                pcm = await decode_to_pcm(await self._messenger.download(file_id))
            except AppError:
                await self._messenger.send_text(chat_id, "Не удалось прочитать голосовое.")
                return
            heard = (await self._stt.transcribe(pcm)).text
            if not heard:
                await self._messenger.send_text(chat_id, "Не расслышал вопрос, повторите.")
                return
            try:
                result = await self._dialog.answer(self._session(chat_id), heard)
            except LLMError:
                log.exception("telegram_llm_failed", stage="telegram")
                await self._messenger.send_text(chat_id, LLM_UNAVAILABLE)
                return
            await self._messenger.send_text(
                chat_id, format_answer(result.answer, heard, with_sources=False)
            )
            spoken = normalize_for_tts(result.answer.text)
            if spoken and self._voice_reply:
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
