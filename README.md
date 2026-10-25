# Голосовой RAG-ассистент

Отвечает на вопросы по корпусу документов (PDF, Markdown) текстом и голосом со ссылкой на источник.
Каналы: REST API, WebSocket (голос в реальном времени), Telegram-бот.

> Статус: в разработке. Требования — [`docs/spec.org`](docs/spec.org).

## Запуск

```bash
cp .env.example .env   # заполнить ключи
docker compose up -d
```

## Структура

| Папка | Назначение |
|-------|------------|
| `frontend/` | Веб-клиент для голоса |
| `services/api/` | Backend: FastAPI, RAG, STT/TTS, Telegram webhook |
| `contracts/` | OpenAPI, события WebSocket, примеры ответов |
| `docs/` | ТЗ и документация |
| `data/` | Демо-документы |

## Метрики

Будут добавлены по результатам eval (recall@5, MRR, p95 задержки до первого звука).
