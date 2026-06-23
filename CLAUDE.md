# CLAUDE.md

Общие правила проекта — в AGENTS.md (единый источник для всех агентов):

@AGENTS.md

## Специфика Claude Code

- Skills — `.agents/skills/` (`.claude/skills` — симлинк на неё):
  - `run-eval` — прогон eval поиска/латентности и сравнение с прошлыми результатами;
  - `add-adapter` — добавление новой реализации адаптера (эмбеддер, хранилище, LLM, STT, TTS) за Protocol;
  - `check-spec` — сверка реализации с требованиями FR/NFR из ТЗ.
- Субагенты — `.claude/agents/`:
  - `rag-reviewer` — ревью изменений на соответствие ТЗ и правилам AGENTS.md;
  - `eval-analyst` — анализ результатов eval и разбор промахов поиска;
  - `spec-navigator` — поиск нужных требований в `docs/spec.org`.
- Перед завершением задачи с кодом (в `services/api/`): `ruff check`, `mypy app/`, `pytest` по затронутым модулям.
