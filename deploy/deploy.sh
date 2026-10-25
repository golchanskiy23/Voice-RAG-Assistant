#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
compose=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)

git pull --ff-only
"${compose[@]}" up -d --build --wait --remove-orphans
"${compose[@]}" ps
docker image prune -f >/dev/null
curl -fsS "https://$(grep -E '^DOMAIN=' .env | cut -d= -f2)/health" && echo
