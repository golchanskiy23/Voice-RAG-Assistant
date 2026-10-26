# HTTP API

Полная схема — OpenAPI FastAPI (`/docs` локально). Здесь — изменения, о которых должны знать клиенты.

## `POST /ask`, `POST /ask/stream`

`{question, session_id?}`. С `session_id` вопрос идёт через историю диалога (Redis, TTL 24 ч, последние
`DIALOG_MAX_TURNS` реплик): уточняющий вопрос переформулируется в самостоятельный перед поиском (FR-4.2).
`session_id` пространство имён `api:`; история Telegram — `tg:<chat_id>`, не пересекаются.
Сейчас история учитывается только в `/ask`; `/ask/stream` и `/ws/voice` отвечают без истории.

`/ask` при недоступном LLM (после ретраев и fallback) → 503 `llm_unavailable`.

## `GET /documents`, `DELETE /documents/{id}`

С `X-API-Key`. `GET` → `{documents: [{id, filename, mime, pages, chunks}]}` в порядке загрузки.
`DELETE` удаляет документ и чанки из Postgres и Qdrant → 204; неизвестный `id` → 404 `not_found`.

## `POST /telegram/webhook`

Только для Telegram. Заголовок `X-Telegram-Bot-Api-Secret-Token` обязан совпасть с `TELEGRAM_SECRET`, иначе 401.
Тело — `Update` Bot API. Ответ `{"ok": true}` сразу; обработка — в фоне. Повтор того же `update_id` в течение
часа не обрабатывается (дедуп в Redis). Без `TELEGRAM_TOKEN` или при `TELEGRAM_MODE=polling` — 404.

## `/search`

`{query, k, doc_ids?, mode?: vector|bm25|hybrid, store?, rerank?}` → у чанка `source` (`vector`/`bm25`/`hybrid`)
и `found_by`; в ответе `mode`, `reranked`.

## `/ready`

Проверки: `postgres`, `qdrant`, `embedder`, `llm`, `stt`, `tts`, `redis` (или `dialog` без Redis),
`reranker` и `telegram` — если включены.
