# Деплой на VPS

Цель: бот и API работают 24/7 по HTTPS, после перезагрузки VPS всё поднимается само (NFR-9).

## Сервер

- VPS: 2 vCPU, 4 ГБ RAM, 30 ГБ диска, Ubuntu 24.04. Без локальных моделей хватит 2 ГБ.
- Домен или поддомен с A-записью на IP сервера (`rag.example.com`) — нужен для Let's Encrypt и вебхука Telegram.

```bash
# под root, один раз
adduser deploy && usermod -aG sudo deploy
mkdir -p /home/deploy/.ssh && cp ~/.ssh/authorized_keys /home/deploy/.ssh/ && chown -R deploy: /home/deploy/.ssh
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/; s/^#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart ssh
ufw default deny incoming && ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443 && ufw enable
curl -fsSL https://get.docker.com | sh && usermod -aG docker deploy
systemctl enable docker          # docker стартует при загрузке, контейнеры с restart: always — за ним
```

Порты Postgres, Qdrant, Redis и API наружу не открыты (`ports: !reset []` в prod-оверлее): снаружи только Caddy на 80/443.
Docker обходит ufw для опубликованных портов — поэтому важно не публиковать их, а не надеяться на ufw.

## Первый запуск

```bash
# под deploy
git clone https://github.com/golchanskiy23/Voice-RAG-Assistant.git && cd Voice-RAG-Assistant
cp .env.example .env && chmod 600 .env
```

В `.env` обязательно:

| Переменная | Значение |
|---|---|
| `DOMAIN`, `ACME_EMAIL` | домен и почта для Let's Encrypt |
| `POSTGRES_PASSWORD`, `ADMIN_API_KEY` | случайные: `openssl rand -hex 24` |
| `TELEGRAM_TOKEN` | от @BotFather |
| `TELEGRAM_SECRET` | `openssl rand -hex 32` — Telegram шлёт его в `X-Telegram-Bot-Api-Secret-Token` |
| `TELEGRAM_WHITELIST` | chat_id через запятую; пусто — бот открыт всем (с rate limit) |
| `LLM_PROVIDER`, `OPENAI_API_KEY` / `GIGACHAT_CREDENTIALS`, `LLM_FALLBACK` | реальный LLM |
| `EMBEDDER` | `openai` без локальных моделей; `e5` — с `WITH_LOCAL_MODELS=1` |
| `STT`, `TTS`, `VAD` | `faster-whisper`, `silero`, `silero` — с `WITH_LOCAL_MODELS=1`; иначе голос не распознаётся |

```bash
deploy/deploy.sh      # git pull → build → up --wait → проверка https://$DOMAIN/health
```

Caddy сам получает и продлевает сертификат. При старте api регистрирует вебхук
`https://$DOMAIN/telegram/webhook` с секретом (`setWebhook`, `allowed_updates=["message"]`).
Проверка: `curl https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo`.

Загрузка документов: `curl -X POST https://$DOMAIN/documents -H "X-API-Key: $ADMIN_API_KEY" -F file=@doc.pdf`.

## Память (NFR-8, ≤ 4 ГБ)

Лимиты в `docker-compose.prod.yml`: api 2.5 ГБ (`API_MEMORY`), Postgres 512 МБ, Qdrant 512 МБ, Redis 128 МБ
(`maxmemory 64mb`, `volatile-lru` — вытесняются ключи с TTL), Caddy 64 МБ.
С локальными моделями в api: whisper `small` int8 (~0.5 ГБ), Silero TTS (~0.3 ГБ), e5-base (~1.1 ГБ) + torch.
Не влезает — `EMBEDDER=openai` (минус e5), reranker выключен. Факт — `docker stats` после прогрева; не измерен.
Веса моделей кэшируются в volume `models` (`HF_HOME`, `TORCH_HOME`) и не скачиваются при пересборке.

## Логи

- Все сервисы пишут в stdout; драйвер `json-file` с ротацией: 5 файлов × 10 МБ на контейнер.
- api — JSON (structlog): `request_id`, `stage`, латентность по этапам, токены, провайдер, версия промпта;
  тексты вопросов — только на `LOG_LEVEL=DEBUG`.
- Смотреть: `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api | jq 'select(.event=="rag_answer")'`.
- `/ready`, `/docs` и `/openapi.json` снаружи закрыты в Caddy (404).

## Бэкап Postgres

```bash
crontab -e
# каждый день в 03:15, хранить 14 дней
15 3 * * * /home/deploy/Voice-RAG-Assistant/deploy/backup.sh >> /home/deploy/backups/backup.log 2>&1
```

Восстановление: `gunzip -c postgres-….sql.gz | docker compose … exec -T postgres psql -U rag -d rag`.
Qdrant восстанавливается переиндексацией из Postgres (тексты чанков там) — отдельный бэкап не нужен.
Бэкапы лучше копировать за пределы VPS (`rclone`/`rsync` в объектное хранилище).

## Проверка автоподъёма (NFR-9)

`sudo reboot` → через минуту `curl https://$DOMAIN/health` и сообщение боту. Не проверено: VPS в этой работе не было.
