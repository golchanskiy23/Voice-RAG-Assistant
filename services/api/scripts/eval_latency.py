import argparse
import asyncio
import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

from voice_client import converse  # noqa: E402

QUESTIONS = API_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = API_DIR / "eval" / "results"
STAGES = [
    "stt",
    "retrieve",
    "first_token",
    "first_sentence",
    "first_synthesized",
    "first_audio",
    "total",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 1)


def load_requests(audio_dir: Path | None, limit: int) -> list[dict[str, Any]]:
    if audio_dir is not None:
        files = sorted(
            p for p in audio_dir.iterdir() if p.suffix.lower() in {".wav", ".ogg", ".mp3"}
        )
        return [{"audio": path, "id": path.stem} for path in files][:limit]
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    requests = [{"text": row["question"], "id": row["id"]} for row in rows]
    while len(requests) < limit:
        requests.extend(requests[: limit - len(requests)])
    return requests[:limit]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    requests = load_requests(args.audio_dir, args.n)
    marks: dict[str, list[float]] = {stage: [] for stage in [*STAGES, "client_first_audio"]}
    errors = 0
    server: dict[str, Any] = {}
    for request in requests:
        result = await converse(
            args.url,
            audio=request.get("audio"),
            text=request.get("text"),
            realtime=args.realtime,
            out=None,
            quiet=True,
        )
        server = result["ready"]
        if "done" not in result:
            errors += 1
            continue
        for stage, value in result["done"]["latency"].items():
            marks.setdefault(stage, []).append(value)
        if result["client_first_audio_ms"] is not None:
            marks["client_first_audio"].append(result["client_first_audio_ms"])
    table = {
        stage: {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "mean": round(statistics.fmean(values), 1),
        }
        for stage, values in marks.items()
        if values
    }
    return {
        "config": args.config,
        "date": datetime.date.today().isoformat(),
        "requests": len(requests),
        "errors": errors,
        "input": "audio" if args.audio_dir else "text",
        "server": {"stt": server.get("stt"), "tts": server.get("tts")},
        "note": args.note,
        "latency_ms": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="p50/p95 по этапам голосового ответа через /ws/voice"
    )
    parser.add_argument("--config", required=True, help="имя конфигурации для файла результата")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--audio-dir", type=Path, help="записи вопросов; без него — вопросы текстом из eval"
    )
    parser.add_argument("--realtime", action="store_true", help="слать аудио в реальном времени")
    parser.add_argument("--url", default="ws://localhost:8000/ws/voice")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(f"{'этап':22} {'p50, мс':>10} {'p95, мс':>10}")
    for stage, values in report["latency_ms"].items():
        print(f"{stage:22} {values['p50']:10.1f} {values['p95']:10.1f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{report['date']}_latency_{args.config}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"сохранено: {path.relative_to(API_DIR)}, ошибок: {report['errors']}")


if __name__ == "__main__":
    main()
