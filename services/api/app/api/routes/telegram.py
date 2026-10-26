import asyncio
import hmac
from typing import Any, Protocol

import structlog
from fastapi import APIRouter, Request

from app.adapters.dialog.base import DialogStore
from app.core.errors import AppError

log = structlog.get_logger()

router = APIRouter(tags=["telegram"])
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"
DEDUP_TTL_SECONDS = 3600


class UpdateProcessor(Protocol):
    async def process(self, payload: dict[str, Any]) -> None: ...


background: set[asyncio.Task[None]] = set()


async def process_safely(processor: UpdateProcessor, payload: dict[str, Any]) -> None:
    try:
        await processor.process(payload)
    except Exception:
        log.exception(
            "telegram_update_failed", stage="telegram", update_id=payload.get("update_id")
        )


@router.post("/telegram/webhook")
async def webhook(request: Request) -> dict[str, bool]:
    state = request.app.state
    processor: UpdateProcessor | None = state.telegram
    secret = state.settings.telegram_secret
    if processor is None or state.settings.telegram_mode != "webhook" or secret is None:
        raise AppError("not_found", "Telegram-бот не настроен", status_code=404)
    expected: str = secret.get_secret_value()
    received = request.headers.get(SECRET_HEADER, "")
    if not hmac.compare_digest(received.encode(), expected.encode()):
        raise AppError("unauthorized", "Неверный secret token", status_code=401)
    payload: dict[str, Any] = await request.json()
    update_id = payload.get("update_id")
    if not isinstance(update_id, int):
        raise AppError("bad_request", "Нет update_id", status_code=400)
    store: DialogStore = state.dialog_store
    if not await store.first_seen(f"tg-update:{update_id}", DEDUP_TTL_SECONDS):
        log.info("telegram_duplicate", stage="telegram", update_id=update_id)
        return {"ok": True}
    task = asyncio.create_task(process_safely(processor, payload))
    background.add(task)
    task.add_done_callback(background.discard)
    return {"ok": True}
