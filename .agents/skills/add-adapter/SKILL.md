---
name: add-adapter
description: Добавить новую реализацию адаптера (эмбеддер, векторное хранилище, LLM-клиент, STT, TTS) за существующим Protocol в app/adapters/ и подключить её через конфиг. Использовать, когда нужно добавить провайдера/модель, например BGE-M3, GigaChat, pgvector.
---

# Новый адаптер

Область применения: `services/api/` (пути ниже — относительно неё).

## Шаги

1. Найди Protocol в `app/adapters/<тип>/base.py` (или аналоге). Если его нет — сначала создай Protocol по ТЗ:
   - `Embedder`: `async embed_documents(texts) -> list[list[float]]`, `async embed_query(text)`, `dim: int` (FR-2.6);
   - `VectorStore`: `upsert`, `search(vector, k, doc_ids)`, `delete_document` (FR-2.1, FR-1.6);
   - `LLMClient`: `complete`, `stream` (FR-3.1);
   - `STT` / `TTS`: FR-5.1 / FR-5.4.
2. Создай реализацию в `app/adapters/<тип>/<имя>.py`. Без LangChain/LlamaIndex — только прямые SDK.
3. Локальный инференс (PyTorch, faster-whisper, Silero) — загрузка модели один раз, вызовы через `asyncio.to_thread`.
4. Зарегистрируй реализацию в фабрике, выбор по env (`EMBEDDER`, `VECTOR_STORE`, `LLM_PROVIDER`, …); добавь переменную в `app/core/config.py` и в корневой `.env.example`.
5. Для эмбеддеров: учти префиксы (`query:` / `passage:` для e5), нормализацию, отдельную Qdrant-коллекцию `chunks_<embedder>`.
6. Тесты: unit-тест соответствия Protocol (можно с моком SDK) + один интеграционный, помеченный `@pytest.mark.slow`, если нужна реальная модель.
7. Проверь: `uv run ruff check . && uv run mypy app/ && uv run pytest`.
8. Если адаптер влияет на поиск — предложи прогнать skill `run-eval`.
