import argparse
import os
import sys
import time
from pathlib import Path

import httpx2

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.api.security import API_KEY_HEADER  # noqa: E402
from app.services.ingestion import EXTENSIONS  # noqa: E402

UPLOAD_TIMEOUT_SECONDS = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Индексация папки с PDF и Markdown через API")
    parser.add_argument("path", type=Path, help="папка (обходится рекурсивно) или файл")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("ADMIN_API_KEY", "dev-admin-key"))
    return parser.parse_args()


def find_files(path: Path) -> list[Path]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    return [p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS]


def main() -> int:
    args = parse_args()
    files = find_files(args.path)
    if not files:
        print(f"нет PDF/Markdown в {args.path}", file=sys.stderr)
        return 1
    failed = 0
    chunks = 0
    started = time.perf_counter()
    headers = {API_KEY_HEADER: args.api_key}
    with httpx2.Client(base_url=args.api_url, timeout=UPLOAD_TIMEOUT_SECONDS) as client:
        for path in files:
            response = client.post(
                "/documents", headers=headers, files={"file": (path.name, path.read_bytes())}
            )
            if response.is_success:
                body = response.json()
                chunks += body["chunks"]
                print(f"{body['status']:>8}  {body['chunks']:>4} фрагм.  {path}")
            else:
                failed += 1
                print(f"{'ошибка':>8}  {response.status_code}  {path}: {response.text}")
    seconds = time.perf_counter() - started
    print(f"файлов: {len(files)}, ошибок: {failed}, фрагментов: {chunks}, {seconds:.1f} с")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
