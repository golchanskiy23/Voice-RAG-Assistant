# Голосовой RAG-ассистент

Ассистент отвечает на вопросы по корпусу документов (PDF, Markdown) текстом и голосом, всегда со ссылкой на источник:
файл, страницу или раздел и цитату. Если ответа в документах нет, он прямо говорит об этом и не выдумывает.

RAG реализован вручную, без LangChain, LlamaIndex и подобных фреймворков. Качество поиска и задержка голосового
ответа измеряются воспроизводимым eval.

> **Статус:** в разработке — готовы каркас сервиса, инфраструктура, загрузка и индексация документов, векторный поиск
> top-k в Qdrant и pgvector за общим интерфейсом, ответ LLM (OpenAI / GigaChat с fallback) со ссылкой на источник
> и SSE-стримингом; гибридный поиск и голос впереди.
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
# {"id":"...","chunks":3,"pages":null,"status":"created"}
```

## API

| Метод | Путь | Назначение | Статус |
|-------|------|------------|--------|
| GET | `/health` | Процесс жив | ✅ |
| GET | `/ready` | Postgres, Qdrant, эмбеддер и LLM доступны, иначе 503 | ✅ |
| POST | `/documents` | Загрузить PDF/Markdown до 20 МБ (API-ключ в `X-API-Key`): парсинг, чанкинг, эмбеддинги, запись чанков в Postgres (+ pgvector) и Qdrant; тот же файл → тот же `id`, новая версия с тем же именем заменяет старую | ✅ |
| POST | `/ask` | Ответ целиком: `{answer, sources: [{filename, page, section, quote}], confidence, latency}` | ✅ |
| POST | `/ask/stream` | SSE: `token` (текст ответа), `sources` (источники и confidence), `done` (latency), `error` | ✅ |
| GET | `/documents` | Список документов | план |
| DELETE | `/documents/{id}` | Удалить документ и его чанки | план |
| POST | `/search` | Поиск top-k чанков: `{query, k, doc_ids?, mode, store?}` → `{chunks: [{id, document_id, filename, ord, text, page, section, score, source}], store, latency}` | ✅ `mode=vector` |
| WS | `/ws/voice` | Аудио → `partial_transcript`, `final_transcript`, `answer_text`, `audio_chunk`, `done` | план |
| POST | `/telegram/webhook` | Обновления Telegram | план |

Ошибки возвращаются в едином формате: `{"error": {"code", "message", "request_id"}}`.
Каждый ответ содержит заголовок `X-Request-ID`.

## Конфигурация

Переменные читаются из окружения (в compose — из `.env` в корне):

| Переменная | По умолчанию в compose | Назначение |
|------------|------------------------|------------|
| `ADMIN_API_KEY` | `dev-admin-key` | Ключ для `/documents` |
| `LLM_PROVIDER` | `stub` | LLM: `openai`, `gigachat` или `stub` — без сети, цитирует первый найденный чанк (для разработки) |
| `LLM_FALLBACK` | — | Второй провайдер: при ошибке первого (после ретраев) запрос уходит ему |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Модель OpenAI для ответа (нужна поддержка strict JSON Schema) |
| `GIGACHAT_CREDENTIALS` | — | Authorization key GigaChat (base64 `client_id:client_secret`), обязателен для `gigachat` |
| `GIGACHAT_SCOPE`, `GIGACHAT_MODEL` | `GIGACHAT_API_PERS`, `GigaChat-2` | Scope OAuth и модель GigaChat |
| `GIGACHAT_VERIFY_SSL`, `GIGACHAT_CA_BUNDLE_FILE` | `true`, — | Проверка TLS; API подписан НУЦ Минцифры — нужен его корневой сертификат |
| `LLM_TIMEOUT`, `LLM_MAX_TOKENS` | `30`, `512` | Таймаут запроса к LLM, с; потолок токенов ответа |
| `PROMPT_VERSION` | `v1` | Версия промпта из `services/api/prompts/answer/<версия>/` |
| `RAG_TOP_K` | `5` | Сколько чанков искать для ответа |
| `RAG_MIN_SCORE` | `0.0` | Порог score: если ни один чанк не прошёл — отказ без вызова LLM |
| `RAG_CONTEXT_TOKENS` | `3000` | Бюджет токенов на чанки в промпте; не поместившиеся отбрасываются с конца |
| `LOG_LEVEL` | `INFO` | Уровень логов; текст вопроса пишется только на `DEBUG` |
| `EMBEDDER` | `hashing` | Эмбеддер: `hashing` — feature hashing без модели, для разработки; `openai` — OpenAI embeddings |
| `OPENAI_API_KEY` | — | Обязателен при `EMBEDDER=openai` и LLM `openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Модель OpenAI; размерность приводится к 1024 параметром `dimensions` |
| `EMBEDDING_BATCH_SIZE` | `128` | Текстов в одном запросе к эмбеддеру |
| `EMBEDDING_CACHE_SIZE` | `10000` | LRU-кэш эмбеддингов по SHA-256 текста; `0` — выключен |
| `VECTOR_STORE` | `qdrant` | Хранилище для `/search` по умолчанию: `qdrant` или `pgvector`; в запросе переопределяется полем `store` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `300`, `45` | Размер чанка и перекрытие в токенах эмбеддера (см. «Обработка документов») |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `rag` | Учётные данные Postgres |
| `API_PORT` | `8000` | Порт api на хосте |
| `POSTGRES_PORT`, `QDRANT_PORT` | `5433`, `6334` | Порты Postgres и Qdrant на `127.0.0.1` |

`DATABASE_URL` и `QDRANT_URL` compose собирает сам. При запуске сервиса без compose они обязательны.
Полный список будущих переменных — в [`docs/spec.md`](docs/spec.md), раздел 9.

## Обработка документов

`POST /documents` → парсинг → чанкинг → эмбеддинги → одна транзакция в Postgres (`documents` + `chunks`).

- **PDF** — свой извлекатель на pikepdf (из проекта PDF Cloner): декодирование через `ToUnicode` и ширины глифов,
  текстовые матрицы, Form XObject, WinAnsi для шрифтов без `ToUnicode`. Строки собираются по координатам,
  удаляются колонтитулы (повторяются на ≥ 60% страниц) и номера страниц, склеиваются переносы (`информацион-` + `ной`),
  абзац, перешедший на следующую страницу, собирается целиком. Заголовки — по размеру шрифта
  (≥ 1.2 × основного), уровень — по рангу размера. Скан без текстового слоя → 422 `no_text` (OCR вне объёма).
  pymupdf не используется: лицензия AGPL.
- **Markdown** — деление по заголовкам (`#` и setext), блоки кода не режутся и `#` внутри них не считается заголовком,
  front matter удаляется.
- **Чанкинг** — рекурсивно: раздел (заголовок) → предложения → окна слов для сверхдлинного предложения. Чанки не
  пересекают границу раздела. Перекрытие — целыми предложениями с конца предыдущего чанка.
- У чанка: текст, номер страницы (первой), путь раздела (`Регламент > 2. Загрузка`), число токенов, эмбеддинг,
  `metadata.pages` — все страницы, по которым он прошёл.

### Размер чанка: 300 токенов, перекрытие 45 (15%)

- **Верхняя граница — модель.** У multilingual-e5 лимит 512 токенов вместе со служебными токенами и префиксом
  `passage:`; всё длиннее молча отрезается. 300 оставляет запас и подходит и для BGE-M3. Сервис не стартует, если
  `CHUNK_SIZE + 16` больше лимита эмбеддера.
- **Токены считает токенизатор эмбеддера**, а не tiktoken: числа у них разные (русский текст в cl100k почти вдвое
  «дороже», чем в XLM-R у e5), а лимит относится к токенизатору модели.
- **Точность ссылки.** Чем меньше чанк, тем точнее цитата и страница в ответе; 300 токенов ≈ 1–2 абзаца регламента
  или один ответ FAQ — смысловая единица, на которую обычно ссылаются.
- **Контекст LLM.** top-5 × 300 = 1 500 токенов контекста — дёшево и быстро для первого токена (NFR-3).
- **Перекрытие 15%** (в рекомендованном диапазоне 10–20%): мысль, разрезанная границей чанка, остаётся целой хотя бы
  в одном из соседей. Перекрытие — целыми предложениями, без обрезков.
- **Это гипотеза, а не результат.** Значение будет проверено на eval-наборе (этап 2): recall@5 для 150 / 300 / 500
  токенов, итог — в разделе «Оценка качества».

## Векторный поиск

Эмбеддер и хранилище — за `Protocol` (`Embedder`, `VectorStore`: `upsert`, `search(vector, k, document_ids)`,
`delete_documents`), реализация выбирается env.

- **Эмбеддер OpenAI** — батчи по `EMBEDDING_BATCH_SIZE`, ретраи с экспоненциальным backoff и jitter на 429, таймаутах,
  обрыве соединения и 5xx; поверх любого эмбеддера — LRU-кэш по SHA-256 от `модель:размерность:тип:текст`.
- **pgvector** — колонка `chunks.embedding vector(1024)`, индекс HNSW `vector_cosine_ops`, оператор `<=>`,
  score = `1 − cosine distance`. `hnsw.ef_search = 64`, при фильтре по документу — `iterative_scan`, а если индекс
  вернул меньше k — точный проход.
- **Qdrant** — коллекция `chunks_<эмбеддер>`, cosine, HNSW `m=16, ef_construct=64`, `hnsw_ef = 64`; в payload — текст
  и метаданные чанка, payload-индекс по `document_id` для фильтра.
- **Индексация** — при загрузке чанки пишутся в Postgres вместе с эмбеддингом (pgvector) и в Qdrant; транзакция
  Postgres фиксируется только после успешного upsert в Qdrant, при ошибке — откат и 503 `indexing_failed`.
  Точки заменённой версии документа удаляются из Qdrant.

Пример: на один и тот же запрос оба хранилища вернули одинаковый top-5 с одинаковыми score.

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "задержка голосового ответа p95", "k": 5, "store": "pgvector"}'
```

### Замеры: Qdrant и pgvector

```bash
cd services/api
PYTHONPATH=. uv run python scripts/bench_vector_stores.py   # нужен docker compose up
```

20 000 синтетических чанков (по 100 на документ), размерность 1024, эмбеддер `hashing`, 200 запросов, k = 5,
`ef_search = 64`, оба хранилища в compose на одной машине. recall@5 — относительно точного поиска (seq scan
в Postgres); фильтр — по одному документу.

| Хранилище | Индексация 20k чанков | recall@5 | p50, мс | p95, мс | p50 с фильтром, мс | p95 с фильтром, мс |
|-----------|-----------------------|----------|---------|---------|--------------------|--------------------|
| pgvector | 152.5 с | 0.959 | 8.0 | 9.7 | 6.6 | 9.0 |
| Qdrant | 20.1 с upsert + 13.6 с оптимизация | 0.898 | 6.6 | 8.2 | 6.2 | 8.3 |

Совпадение top-5 между хранилищами — 0.909.

- Индексация pgvector медленнее: вставка идёт в уже существующий HNSW-индекс, граф строится по одной точке на каждый
  INSERT. Qdrant принимает точки сразу и строит граф фоном (до статуса `green`).
- Задержка поиска сопоставима, обе укладываются в бюджет NFR на поиск с запасом.
- recall@5 здесь — точность приближённого поиска, а не качество ответа: векторы `hashing` на синтетическом тексте
  плохо кластеризуются, это худший случай для HNSW. Качество поиска на реальном корпусе — в «Оценка качества».

## Генерация ответа

`/ask`: поиск top-k → фильтр по `RAG_MIN_SCORE` → бюджет контекста → промпт → LLM → валидация → источники.

- **Промпт** — `services/api/prompts/answer/v1/` (`system.md`, `user.md`, `retry.md`). Правила: только по контексту,
  чанки пронумерованы `[1]…[k]` с файлом, страницей и разделом, при отсутствии ответа — фраза
  «В документах нет ответа на этот вопрос.». Версия (`answer/v1`) пишется в лог каждого запроса.
- **Структурированный ответ** — strict JSON Schema (`answer`, `sources`, `confidence`) у OpenAI и GigaChat, затем
  Pydantic `LLMAnswer`: номера источников вне `1..k` — ошибка. Невалидный ответ — один повтор с текстом ошибки, затем
  отказ. Ответ без источников тоже превращается в отказ.
- **Источник** — файл, страница, раздел и цитата до 200 символов из процитированного чанка.
- **Стриминг** — модель стримит JSON, сервис на лету вынимает из него значение `answer` и шлёт событиями `token`;
  после конца стрима JSON валидируется и уходит событием `sources`.
- **LLM за интерфейсом** — `LLMClient` (`complete`, `stream`): OpenAI (ретраи с backoff на 429/5xx/таймаутах),
  GigaChat (OAuth-токен обновляет SDK), `FallbackLLMClient` (переключение на второго провайдера; в стриме — только
  до первого токена).
- **Логи** — JSON (structlog) на каждый запрос: `request_id`, `chunk_ids`, `sources`, `provider`, `prompt_version`,
  `prompt_tokens`/`completion_tokens`, `latency_ms` по этапам (`embed`, `search`, `first_token`, `llm`, `total`).

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "Какой максимальный размер файла?"}'
curl -N localhost:8000/ask/stream -H 'Content-Type: application/json' -d '{"question": "Какой максимальный размер файла?"}'
```

Бюджет контекста на этом вопросе (top-5, токенизатор `o200k_base`): system 250 + вопрос 13 + чанки 1 109 + служебные 11
+ ответ до 512 = 1 895 токенов. Качество ответов с реальной LLM (верный документ на 5 вопросах, отказ вне базы) ещё
не измерено: проверено только с `stub` и на тестах с замоканной LLM.

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
