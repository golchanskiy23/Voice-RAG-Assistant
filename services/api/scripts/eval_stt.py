import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.adapters.voice_registry import stt_registry  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.services.audio import decode_to_pcm  # noqa: E402
from app.services.wer import word_errors  # noqa: E402

RESULTS_DIR = API_DIR / "eval" / "results"


async def run(args: argparse.Namespace) -> dict[str, object]:
    settings = Settings(
        database_url="postgresql+asyncpg://unused:unused@localhost/unused",
        qdrant_url="http://localhost:6333",
        admin_api_key="eval",
        stt=args.stt,
        whisper_model=args.model,
        whisper_compute_type=args.compute_type,
    )
    stt = await asyncio.to_thread(stt_registry.create, settings.stt, settings)
    warm_up = getattr(stt, "warm_up", None)
    if warm_up is not None:
        await warm_up()
    rows = []
    errors = words = 0
    audio_total = compute_total = 0.0
    for audio in sorted(
        p for p in args.dir.iterdir() if p.suffix.lower() in {".wav", ".ogg", ".mp3"}
    ):
        reference_path = audio.with_suffix(".txt")
        if not reference_path.exists():
            continue
        pcm = await decode_to_pcm(audio.read_bytes())
        started = time.perf_counter()
        transcript = await stt.transcribe(pcm)
        elapsed = time.perf_counter() - started
        result = word_errors(reference_path.read_text(encoding="utf-8"), transcript.text)
        errors, words = errors + result.errors, words + result.words
        audio_total += transcript.audio_seconds
        compute_total += elapsed
        rows.append(
            {
                "file": audio.name,
                "wer": round(result.wer, 4),
                "rtf": round(elapsed / transcript.audio_seconds, 3),
                "hypothesis": transcript.text,
            }
        )
        print(
            f"{audio.name:30} WER {result.wer:6.1%}  RTF {elapsed / transcript.audio_seconds:5.2f}"
        )
    await stt.close()
    if not rows:
        raise SystemExit("нет пар <имя>.wav + <имя>.txt")
    return {
        "config": f"{args.stt}_{args.model}_{args.compute_type}",
        "date": datetime.date.today().isoformat(),
        "files": len(rows),
        "wer": round(errors / words, 4),
        "rtf": round(compute_total / audio_total, 3),
        "audio_seconds": round(audio_total, 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WER и RTF распознавания на своих записях")
    parser.add_argument("dir", type=Path, help="папка с <имя>.wav|ogg и эталоном <имя>.txt")
    parser.add_argument("--stt", default="faster-whisper")
    parser.add_argument("--model", default="small")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(f"WER {report['wer']:.1%} (корпусный), RTF {report['rtf']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{report['date']}_stt_{report['config']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"сохранено: {path.relative_to(API_DIR)}")


if __name__ == "__main__":
    main()
