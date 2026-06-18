---
name: run-eval
description: Прогнать оценку качества поиска (recall@k, MRR) или латентности (p50/p95) и сравнить с прошлыми результатами в eval/results/. Использовать после любых изменений чанкинга, эмбеддера, поиска, reranker'а, промптов или голосового пайплайна.
---

# Прогон eval

## Шаги

1. Определи, что менялось:
   - чанкинг / эмбеддер / поиск / rerank → `scripts/eval_retrieval.py` (FR-7.2);
   - STT / TTS / стриминг / LLM-провайдер → `scripts/eval_latency.py` (FR-7.3);
   - промпты → `uv run pytest tests/ -k prompt` (регрессия FR-7.5).
2. Убедись, что сервисы подняты: `docker compose ps` (postgres, qdrant, redis).
3. Запусти нужный скрипт с явной конфигурацией, например:
   ```bash
   uv run python scripts/eval_retrieval.py --config hybrid_rerank_e5
   ```
4. Результат должен лечь в `eval/results/<YYYY-MM-DD>_<конфиг>.json` (FR-7.4).
5. Найди предыдущий результат той же метрики в `eval/results/` и построй таблицу было/стало:

   | Конфиг | recall@1 | recall@3 | recall@5 | MRR |
   |--------|----------|----------|----------|-----|

   Для латентности — p50/p95 по этапам (STT, retrieve, rerank, LLM first token, TTS first chunk, total to first audio).
6. Сверь с целями NFR: recall@5 ≥ 0.85 (NFR-1), первый звук p95 ≤ 2.5 с (NFR-2), первый токен p95 ≤ 1.5 с (NFR-3), поиск p95 ≤ 300 мс (NFR-4).

## Правила

- Не подгоняй датасет `eval/questions.jsonl` под результат.
- Если метрика просела — сообщи прямо, с цифрами, и перечисли вопросы, которые стали промахами.
- Для глубокого разбора промахов делегируй субагенту `eval-analyst`.
