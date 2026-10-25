from typing import Any

import structlog
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message, Update

from app.services.telegram import ChatAction, TelegramBot

log = structlog.get_logger()


class AiogramMessenger:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_text(self, chat_id: int, text: str) -> None:
        await self._bot.send_message(chat_id, text)

    async def send_voice(self, chat_id: int, ogg: bytes) -> None:
        await self._bot.send_voice(chat_id, BufferedInputFile(ogg, filename="answer.ogg"))

    async def send_action(self, chat_id: int, action: ChatAction) -> None:
        await self._bot.send_chat_action(chat_id, action)

    async def download(self, file_id: str) -> bytes:
        file = await self._bot.get_file(file_id)
        if file.file_path is None:
            raise FileNotFoundError(file_id)
        buffer = await self._bot.download_file(file.file_path)
        if buffer is None:
            raise FileNotFoundError(file_id)
        return buffer.read()


def build_router(service: TelegramBot) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await service.on_start(message.chat.id)

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        await service.on_reset(message.chat.id)

    @router.message(F.voice)
    async def voice(message: Message) -> None:
        assert message.voice is not None
        user_id = message.from_user.id if message.from_user else message.chat.id
        await service.on_voice(
            message.chat.id, user_id, message.voice.file_id, message.voice.duration
        )

    @router.message(F.text)
    async def text(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id
        await service.on_text(message.chat.id, user_id, message.text or "")

    return router


def create_bot(token: str) -> Bot:
    return Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


class AiogramGateway:
    def __init__(self, bot: Bot, service: TelegramBot) -> None:
        self._bot = bot
        self._dispatcher = Dispatcher()
        self._dispatcher.include_router(build_router(service))

    async def process(self, payload: dict[str, Any]) -> None:
        update = Update.model_validate(payload, context={"bot": self._bot})
        await self._dispatcher.feed_update(self._bot, update)

    async def set_webhook(self, url: str, secret: str) -> None:
        await self._bot.set_webhook(url, secret_token=secret, allowed_updates=["message"])

    async def is_ready(self) -> bool:
        return True

    async def close(self) -> None:
        await self._bot.session.close()
