# Голосовой RAG-ассистент

Ассистент отвечает на вопросы по корпусу документов (PDF, Markdown) текстом и голосом, всегда со ссылкой на источник:
файл, страницу или раздел и цитату. Если ответа в документах нет, он прямо говорит об этом и не выдумывает.

RAG реализован вручную, без LangChain, LlamaIndex и подобных фреймворков. Качество поиска и задержка голосового
ответа измеряются воспроизводимым eval.

> **Статус:** в разработке. Полное техническое задание — [`docs/spec.md`](docs/spec.md).

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
frontend/              # Веб-клиент для голоса
services/api/          # Backend: FastAPI, RAG, голос, Telegram webhook
contracts/             # OpenAPI, события WebSocket, примеры ответов
docs/                  # ТЗ и документация
data/                  # Демо-документы
.agents/skills/        # Навыки AI-агентов
AGENTS.md              # Инструкции для AI-агентов
docker-compose.yml     # Запуск всего решения
.env.example           # Пример конфигурации
```

## Запуск

```bash
cp .env.example .env          # указать OPENAI_API_KEY / GIGACHAT_CREDENTIALS, TELEGRAM_TOKEN, ADMIN_API_KEY
docker compose up -d          # api, postgres (pgvector), qdrant, redis
```

После запуска:

- OpenAPI-документация — http://localhost:8000/docs
- Проверка готовности — `GET /ready`

Индексация папки с документами:

```bash
cd services/api
uv run python scripts/index.py ../../data/
```

## API

| Метод | Путь | Назначение |
|-------|------|------------|
| POST | `/documents` | Загрузить PDF/Markdown (API-ключ) |
| GET | `/documents` | Список документов |
| DELETE | `/documents/{id}` | Удалить документ и его чанки |
| POST | `/search` | Поиск чанков: `{query, k, doc_ids?, mode}` |
| POST | `/ask` | Ответ целиком: `{answer, sources, confidence, latency}` |
| POST | `/ask/stream` | SSE: `token`, `sources`, `done` |
| WS | `/ws/voice` | Аудио → `partial_transcript`, `final_transcript`, `answer_text`, `audio_chunk`, `done` |
| POST | `/telegram/webhook` | Обновления Telegram |
| GET | `/health`, `/ready` | Живость и готовность зависимостей |

Ошибки возвращаются в едином формате: `{"error": {"code", "message", "request_id"}}`.

## Конфигурация

Основные переменные (полный список — в [`docs/spec.md`](docs/spec.md), раздел 9):

| Переменная | Значения | Назначение |
|------------|----------|------------|
| `LLM_PROVIDER`, `LLM_FALLBACK` | `openai`, `gigachat` | Основной и запасной LLM |
| `EMBEDDER` | `openai`, `e5`, `bge-m3` | Модель эмбеддингов |
| `VECTOR_STORE` | `qdrant`, `pgvector` | Векторное хранилище |
| `RERANK_ENABLED` | `true` / `false` | Reranker на top-N |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | токены | Параметры чанкинга |
| `WHISPER_MODEL` | `small`, `medium`, … | Модель распознавания речи |

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
