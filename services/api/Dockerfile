FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ARG WITH_LOCAL_MODELS=0
ARG WITH_STT=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project \
        $([ "$WITH_LOCAL_MODELS" = "1" ] && echo "--group local-models") \
        $([ "$WITH_STT" = "1" ] && echo "--group stt")


FROM python:3.12-slim

ARG WITH_CLAUDE_CODE=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
        $([ "$WITH_CLAUDE_CODE" = "1" ] && echo "curl ca-certificates") \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$WITH_CLAUDE_CODE" = "1" ]; then \
        export HOME=/tmp/claude-install \
        && curl -fsSL https://claude.ai/install.sh | bash \
        && cp -L "$HOME/.local/bin/claude" /usr/local/bin/claude \
        && rm -rf "$HOME"; \
    fi

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /models \
    && chown app:app /models

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app prompts ./prompts
COPY --chown=app:app app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    DISABLE_AUTOUPDATER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

RUN python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]" \
    && chown -R app:app /app/.tiktoken

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
