# Голосовой RAG-ассистент

Ассистент отвечает на вопросы по корпусу документов (PDF, Markdown) текстом и голосом, всегда со ссылкой на источник:
файл, страницу или раздел и цитату. Если ответа в документах нет, он прямо говорит об этом и не выдумывает.

RAG реализован вручную, без LangChain, LlamaIndex и подобных фреймворков. Качество поиска и задержка голосового
ответа измеряются воспроизводимым eval.

> **Статус:** в разработке — готов каркас сервиса, инфраструктура и загрузка документов; поиск, LLM и голос впереди.
> Полное техническое задание — [`docs/spec.md`](docs/spec.md).

## Возможности

- **Загрузка документов** — PDF и Markdown через API или CLI, дедупликация по SHA-256, замена старой версии.
- **Гибридный поиск** — векторный (Qdrant или pgvector) + BM25 (Postgres `tsvector`), слияние через RRF, reranker на cross-encoder.
- **Ответ со ссылками** — LLM (OpenAI или GigaChat с fallback), ответ валидируется Pydantic-схемой, источники — только из найденных чанков.
- **Стриминг** — токены через SSE, голос через WebSocket: звук начинается до окончания генерации.
- **Голос** — STT на faster-whisper + VAD, TTS на Silero по предложениям, прерывание ответа голосом (barge-in).
- **Диалог** — история в Redis, уточняющие вопросы переформулируются с учётом контекста.
- **Telegram-бот** — webhook, текст и голосовые, whitelist и rate limit.
- **Eval** — recall@k и MRR для поиска, p50/p95 задержки по этапам, регрессионные тесты промптов.

## Архитектура

```
 Telegram ──webhook──┐        Веб/CLI-клиент ──WebSocket──┐
                     ▼                                     ▼
              ┌──────────────────── FastAPI (api) ───────────────────┐
              │  bot handlers   /ask  /ask/stream  /search  /ws/voice │
              │        │                     │                │       │
              │        ▼                     ▼                ▼       │
              │   DialogService ──► RAGService ◄──────── VoicePipeline│
              │   (Redis)           │  rewrite → retrieve   STT (whisper)
              │                     │  → rerank → prompt    TTS (silero)
              │                     │  → LLM → validate              │
              │      ┌──────────────┼───────────────┐                │
              │  Embedder      VectorStore      LLMClient            │
              │ (OpenAI/e5/   (Qdrant/pgvector)  (OpenAI/GigaChat)   │
              │  BGE-M3)       + BM25 (Postgres)                     │
              └──────────────────────────────────────────────────────┘
                 │            │             │            │
             PostgreSQL     Qdrant        Redis      внешние LLM API
             (+pgvector)
```

Конвейер ответа: **переформулировка вопроса → гибридный поиск → rerank → промпт с пронумерованными чанками →
LLM → валидация ответа → (TTS)**.

Все внешние зависимости — эмбеддер, векторное хранилище, LLM, STT, TTS — спрятаны за `Protocol`.
Реализация выбирается через env, без изменений кода.

## Стек

| Область | Технологии |
|---------|------------|
| Сервис | Python 3.12, FastAPI, Pydantic v2, pydantic-settings, structlog |
| Хранилища | PostgreSQL + pgvector, Qdrant, Redis |
| Поиск | OpenAI embeddings, multilingual-e5, BGE-M3, bge-reranker (transformers + PyTorch), tiktoken |
| LLM | OpenAI API, GigaChat |
| Голос | faster-whisper, silero-vad, Silero TTS, ffmpeg |
| Каналы | REST, SSE, WebSocket, aiogram 3 (Telegram) |
| Документы | pymupdf, Markdown-парсер |
| Качество | pytest, ruff, mypy, pre-commit |
| Инфраструктура | Docker Compose, Caddy (HTTPS), VPS |

## Структура репозитория

```
services/api/          # Backend: FastAPI-сервис
  app/                 # api, services, adapters, repositories, core
  migrations/          # Alembic
  tests/               # unit и integration (testcontainers)
docs/spec.md           # Техническое задание
.agents/skills/        # Навыки AI-агентов
AGENTS.md              # Инструкции для AI-агентов
docker-compose.yml     # api, migrate, postgres (pgvector), qdrant
.env.example           # Пример конфигурации
```

По мере развития появятся `frontend/` (веб-клиент для голоса), `contracts/` (OpenAPI, события WebSocket) и `data/` (демо-документы).

## Запуск

Нужен Docker с Compose v2.

```bash
docker compose up -d --build --wait
```

Команда соберёт образ, поднимет Postgres и Qdrant, применит миграции (сервис `migrate`) и запустит api.
Без `.env` используются значения для разработки. Для своих значений: `cp .env.example .env` и отредактировать.

```bash
curl localhost:8000/health    # {"status":"ok"}
curl localhost:8000/ready     # {"status":"ready","checks":{"postgres":true,"qdrant":true,"embedder":true,"llm":true}}
```

- OpenAPI-документация — http://localhost:8000/docs
- Остановить: `docker compose down` (данные сохраняются), удалить данные: `docker compose down -v`

Загрузка документа:

```bash
curl -X POST localhost:8000/documents -H 'X-API-Key: dev-admin-key' -F file=@guide.md
# {"id":"...","chunks":0,"status":"created"}
```

## API

| Метод | Путь | Назначение | Статус |
|-------|------|------------|--------|
| GET | `/health` | Процесс жив | ✅ |
| GET | `/ready` | Postgres, Qdrant, эмбеддер и LLM доступны, иначе 503 | ✅ |
| POST | `/documents` | Загрузить PDF/Markdown до 20 МБ, API-ключ в `X-API-Key`; повторный файл → тот же `id` | ✅ без индексации |
| POST | `/ask` | Ответ целиком: `{answer, sources, confidence, latency}` | ✅ заглушка LLM |
| POST | `/ask/stream` | SSE: `token`, `sources`, `done`, `error` | ✅ заглушка LLM |
| GET | `/documents` | Список документов | план |
| DELETE | `/documents/{id}` | Удалить документ и его чанки | план |
| POST | `/search` | Поиск чанков: `{query, k, doc_ids?, mode}` | план |
| WS | `/ws/voice` | Аудио → `partial_transcript`, `final_transcript`, `answer_text`, `audio_chunk`, `done` | план |
| POST | `/telegram/webhook` | Обновления Telegram | план |

Ошибки возвращаются в едином формате: `{"error": {"code", "message", "request_id"}}`.
Каждый ответ содержит заголовок `X-Request-ID`.

## Конфигурация

Переменные читаются из окружения (в compose — из `.env` в корне):

| Переменная | По умолчанию в compose | Назначение |
|------------|------------------------|------------|
| `ADMIN_API_KEY` | `dev-admin-key` | Ключ для `/documents` |
| `LLM_PROVIDER` | `stub` | Реализация LLM (пока только заглушка) |
| `EMBEDDER` | `hashing` | Эмбеддер; `hashing` — feature hashing без модели, для разработки до подключения e5/BGE-M3 |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `rag` | Учётные данные Postgres |
| `API_PORT` | `8000` | Порт api на хосте |
| `POSTGRES_PORT`, `QDRANT_PORT` | `5433`, `6334` | Порты Postgres и Qdrant на `127.0.0.1` |

`DATABASE_URL` и `QDRANT_URL` compose собирает сам. При запуске сервиса без compose они обязательны.
Полный список будущих переменных — в [`docs/spec.md`](docs/spec.md), раздел 9.

## Разработка

Нужен [uv](https://docs.astral.sh/uv/). Команды — из `services/api`:

```bash
uv sync                          # зависимости
uv run pytest                    # unit-тесты
uv run pytest -m integration     # интеграционные: Postgres и Qdrant в testcontainers, нужен Docker
uv run ruff check . && uv run ruff format --check .
uv run mypy app tests migrations
```

Миграции:

```bash
uv run alembic revision -m "описание"                                   # новая миграция
docker compose run --rm migrate                                          # применить в compose
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag uv run alembic upgrade head
```

Pre-commit (из корня репозитория): trailing whitespace, YAML/TOML, приватные ключи, ruff, ruff format, mypy.

```bash
uvx pre-commit install           # хуки на git commit
uvx pre-commit run --all-files
```

## Оценка качества

```bash
cd services/api
uv run python scripts/eval_retrieval.py --config <name>   # recall@1/3/5, MRR
uv run python scripts/eval_latency.py                     # p50/p95 по этапам
```

Датасет: 50 вопросов с эталонными чанками и ответами, из них не меньше 5 — вне базы.
Результаты сохраняются в `services/api/eval/results/`.

| Конфигурация | recall@5 | MRR |
|--------------|----------|-----|
| Векторный поиск (baseline) | — | — |
| Гибрид (BM25 + векторы, RRF) | — | — |
| Гибрид + reranker | — | — |

| Метрика | Цель | Факт |
|---------|------|------|
| recall@5, лучшая конфигурация | ≥ 0.85 | — |
| Задержка до первого звука, p95 | ≤ 2.5 с | — |
| Задержка до первого токена, p95 | ≤ 1.5 с | — |
| Поиск (retrieve + rerank), p95 | ≤ 300 мс | — |
| Выдуманные ответы на вопросы вне базы | 0 | — |

Таблицы заполняются только реально измеренными значениями.

## Этапы

1. **Текстовый RAG** — загрузка документов, векторный поиск, `/ask` со ссылкой на источник, отказ на вопросы вне базы.
2. **Оценка** — eval-датасет, гибридный поиск, reranker, сравнение трёх эмбеддеров.
3. **Голос** — голос → голос через WebSocket, замер p50/p95 до и после оптимизации.
4. **Канал и деплой** — Telegram-бот, диалог в Redis, деплой на VPS по HTTPS.

## Решения

- **Без RAG-фреймворков.** Чанкинг, RRF, rerank и промпты написаны вручную: каждый этап прозрачен и измеряется отдельно.
- **Гибридный поиск.** BM25 ловит точные термины и коды, векторы — перефразировки; RRF объединяет их без подбора весов.
- **Структурированный ответ.** Pydantic-схема с номерами источников отсекает ссылки на несуществующие чанки, а порог score — ответы без опоры на документы.
- **Синтез по предложениям.** TTS запускается на первом законченном предложении, а не после всего ответа, — это главный рычаг задержки до первого звука.
- **Инференс вне event loop.** Модели вызываются через executor, поэтому параллельные запросы не блокируют друг друга.
- **Только инференс.** Модели не обучаются и не дообучаются; качество поднимается поиском, rerank и промптами.
