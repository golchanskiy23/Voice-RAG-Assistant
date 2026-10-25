#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
backup_dir="${BACKUP_DIR:-$HOME/backups/voice-rag}"
keep_days="${BACKUP_KEEP_DAYS:-14}"
mkdir -p "$backup_dir"

file="$backup_dir/postgres-$(date +%Y%m%d-%H%M%S).sql.gz"
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner' | gzip > "$file.tmp"
mv "$file.tmp" "$file"
find "$backup_dir" -name 'postgres-*.sql.gz' -mtime +"$keep_days" -delete
echo "backup: $file ($(du -h "$file" | cut -f1))"
