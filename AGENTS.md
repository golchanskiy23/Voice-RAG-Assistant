# AGENTS.md — Голосовой RAG-ассистент

Инструкции для любых AI-агентов (Claude Code, Codex, Cursor и др.), работающих в этом репозитории.

## Источник правды

- Полное ТЗ — `docs/spec.org` (FR-*, NFR-*, архитектура, данные, API, env, этапы и критерии приёмки).
- Перед реализацией фичи найди соответствующие требования (`FR-x.y` / `NFR-n`) и ссылайся на них в коммитах и PR.
- Если код расходится с ТЗ — не меняй ТЗ молча: укажи расхождение и спроси.

## Проект в двух словах

Ассистент отвечает на вопросы по корпусу документов (PDF, Markdown) текстом и голосом со ссылкой на источник.
Каналы: REST API, WebSocket (`/ws/voice`), Telegram-бот (webhook).
Цель — портфолио AI-инженера с измеренным качеством поиска (recall@5) и задержкой голосового ответа (p95).

## Жёсткие ограничения

- **Без LangChain, LlamaIndex и подобных фреймворков.** RAG (чанкинг, поиск, RRF, rerank, промпты) пишется руками.
- **Без обучения/дообучения моделей** — только инференс готовых.
- Не входит: веб-админка, мультитенантность, авторизация (кроме whitelist бота), OCR сканов.
- Секреты только в env; `.env` в `.gitignore`, рядом `.env.example`.

## Стек

Python 3.12, FastAPI, Pydantic v2 + pydantic-settings, asyncpg/SQLAlchemy, PostgreSQL + pgvector, Qdrant,
Redis (async), OpenAI SDK, GigaChat SDK, transformers + PyTorch (e5, BGE-M3, bge-reranker), faster-whisper,
silero-vad, Silero TTS, ffmpeg, aiogram 3, tiktoken, pymupdf, structlog, pytest, ruff, mypy, Docker Compose, Caddy.

## Структура репозитория

```
frontend/              # Frontend-приложение (веб-клиент для голоса, сценарий С3)
services/              # Backend-сервисы
  api/                 # FastAPI: REST, WebSocket, Telegram webhook
contracts/             # Согласованные API, схемы и примеры ответов (OpenAPI, события WS)
docs/                  # Требования (spec.org) и общая документация
data/                  # Подготовленные демо-данные
.agents/skills/        # Навыки с указанной областью применения
AGENTS.md              # Общие инструкции проекта
README.md              # Описание и общий запуск
docker-compose.yml     # Запуск всего решения
.env.example           # Пример конфигурации без секретов
```

Структура `services/api/`:

```
app/
  api/        роутеры, схемы запросов/ответов
  services/   rag, dialog, voice, ingestion
  adapters/   embedders, vector_stores, llm, stt, tts — каждый: Protocol + реализации
  core/       конфиг (pydantic-settings), логирование, ошибки
prompts/      шаблоны промптов с версией
scripts/      index.py, eval_retrieval.py, eval_latency.py
eval/         questions.jsonl, results/<дата>_<конфиг>.json
tests/
```

- Изменение API или событий WebSocket — сначала обнови `contracts/`, затем код сервиса и клиента.
- Новый навык — в `.agents/skills/<имя>/SKILL.md`, в начале указать область применения.

## Правила кода

- Все внешние зависимости (LLM, эмбеддер, хранилище, STT, TTS) — за `typing.Protocol`; выбор реализации — конфигом.
- Блокирующие вызовы (инференс моделей, синхронные SDK) — через `asyncio.to_thread` / executor. Event loop не блокировать.
- Клиенты БД и модели поднимаются один раз в `lifespan`, прокидываются через `Depends`.
- Ответ LLM валидируется Pydantic-схемой (`answer`, `sources`, `confidence`); номера источников вне контекста — ошибка.
- Промпты — в `prompts/`, с версией; версия пишется в лог каждого запроса.
- Логи — структурированный JSON (structlog): `request_id`, этап, длительность, токены, провайдер, версия промпта.
  Текст вопросов/персональных данных — не на уровне INFO.
- Ошибки API — единый формат `{"error": {"code", "message", "request_id"}}`.
- Типизация обязательна: `mypy app/` без ошибок, `ruff check` и `ruff format` чистые.

## Команды

```bash
docker compose up -d                 # из корня: поднять api, postgres(pgvector), qdrant, redis
cd services/api                      # остальные команды — отсюда
uv run pytest                        # тесты
uv run ruff check . && uv run ruff format --check .
uv run mypy app/
uv run python scripts/index.py ../../data/  # индексация папки
uv run python scripts/eval_retrieval.py --config <name>   # recall@1/3/5, MRR
uv run python scripts/eval_latency.py                     # p50/p95 по этапам
```

(Команды актуализировать по мере появления кода.)

## Тесты и метрики

- Unit: чанкинг, RRF, валидация ответа LLM. Интеграционные: `/ask` с замоканным LLM. Покрытие ключевых модулей ≥ 60%.
- Регрессия промптов (FR-7.5): формат ответа, наличие источника, отказ на вопросы вне базы.
- Любое изменение поиска/чанкинга/эмбеддера → прогон `eval_retrieval.py` и сравнение с предыдущим результатом в `eval/results/`.
- Цифры в README — только реально измеренные; не достигли цели — пишем факт и причину.

## Git

- Коммиты небольшие, сообщение на русском или английском, со ссылкой на требование: `feat(search): RRF hybrid (FR-2.3)`.
- Не коммитить `.env`, веса моделей, большие бинарники. В `data/` — только небольшие демо-документы, которые можно публиковать.
