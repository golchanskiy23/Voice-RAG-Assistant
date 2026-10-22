import argparse
import asyncio
import datetime
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from alembic import command
from alembic.config import Config
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.adapters.embedders.base import Embedder  # noqa: E402
from app.adapters.embedders.registry import build_embedder  # noqa: E402
from app.adapters.rerankers.registry import build_reranker  # noqa: E402
from app.adapters.vector_stores.base import SearchHit, StoreName, VectorStore  # noqa: E402
from app.adapters.vector_stores.pgvector import PgVectorStore  # noqa: E402
from app.adapters.vector_stores.qdrant import QdrantVectorStore  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.database import Database  # noqa: E402
from app.core.qdrant import Qdrant  # noqa: E402
from app.repositories.documents import SqlDocumentRepository  # noqa: E402
from app.repositories.fulltext import PostgresBM25  # noqa: E402
from app.services.chunking import Chunker  # noqa: E402
from app.services.ingestion import EXTENSIONS, IngestionService  # noqa: E402
from app.services.search import SearchDefaults, SearchMode, SearchService  # noqa: E402

QUESTIONS = API_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = API_DIR / "eval" / "results"
DEFAULT_DATABASE_URL = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
EVAL_DATABASE = "rag_eval"
KS = (1, 3, 5)
MRR_DEPTH = 10
SKIPPED_FILES = {"README.md"}


@dataclass(frozen=True)
class Evidence:
    file: str
    contains: str


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    type: str
    expected: tuple[Evidence, ...]

    @property
    def in_scope(self) -> bool:
        return bool(self.expected)


@dataclass
class QuestionResult:
    id: str
    type: str
    first_relevant_rank: int | None
    recall: dict[str, float]
    top1_score: float | None
    top_files: list[str]
    latency_ms: float


@dataclass
class RunResult:
    config: str
    date: str
    settings: dict[str, Any]
    indexing: dict[str, float]
    metrics: dict[str, float]
    by_type: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    questions: list[QuestionResult] = field(default_factory=list)


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        expected = tuple(Evidence(e["file"], e["contains"]) for e in row["expected"])
        questions.append(Question(row["id"], row["question"], row["type"], expected))
    return questions


def is_relevant(hit: SearchHit, evidence: Evidence) -> bool:
    return hit.filename == evidence.file and normalize(evidence.contains) in normalize(hit.text)


def score_question(question: Question, hits: list[SearchHit], latency: float) -> QuestionResult:
    first = next(
        (
            rank
            for rank, hit in enumerate(hits, 1)
            if any(is_relevant(hit, e) for e in question.expected)
        ),
        None,
    )
    recall = (
        {
            f"recall@{k}": sum(any(is_relevant(h, e) for h in hits[:k]) for e in question.expected)
            / len(question.expected)
            for k in KS
        }
        if question.in_scope
        else {}
    )
    return QuestionResult(
        id=question.id,
        type=question.type,
        first_relevant_rank=first,
        recall=recall,
        top1_score=round(hits[0].score, 6) if hits else None,
        top_files=[h.filename for h in hits[:5]],
        latency_ms=round(latency, 2),
    )


def aggregate(results: list[QuestionResult]) -> dict[str, float]:
    scoped = [r for r in results if r.recall]
    if not scoped:
        return {}
    metrics = {
        name: round(statistics.fmean(r.recall[name] for r in scoped), 4)
        for name in (f"recall@{k}" for k in KS)
    }
    metrics["mrr"] = round(
        statistics.fmean(
            1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in scoped
        ),
        4,
    )
    metrics["questions"] = len(scoped)
    return metrics


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 2)


def eval_database_url(base: str) -> str:
    return make_url(base).set(database=EVAL_DATABASE).render_as_string(hide_password=False)


async def ensure_database(base: str) -> None:
    engine = create_async_engine(base, isolation_level="AUTOCOMMIT")
    async with engine.connect() as connection:
        exists = await connection.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": EVAL_DATABASE}
        )
        if not exists:
            await connection.execute(text(f'CREATE DATABASE "{EVAL_DATABASE}"'))
    await engine.dispose()


def migrate(database_url: str) -> None:
    os.environ["DATABASE_URL"] = database_url
    command.upgrade(Config(str(API_DIR / "alembic.ini")), "head")


def corpus_files(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS and p.name not in SKIPPED_FILES
    )


async def index_corpus(
    data_dir: Path,
    database: Database,
    embedder: Embedder,
    chunker: Chunker,
    qdrant: QdrantVectorStore,
) -> dict[str, float]:
    async with database.sessionmaker() as session:
        await session.execute(text("TRUNCATE documents CASCADE"))
        await session.commit()
    files = await asyncio.to_thread(corpus_files, data_dir)
    started = time.perf_counter()
    chunks = 0
    for path in files:
        async with database.sessionmaker() as session:
            service = IngestionService(SqlDocumentRepository(session), embedder, chunker, qdrant)
            result = await service.upload(path.name, path.read_bytes())
            chunks += result.document.chunks
    return {
        "documents": len(files),
        "chunks": chunks,
        "seconds": round(time.perf_counter() - started, 2),
    }


async def run_mode(
    service: SearchService, questions: list[Question], mode: SearchMode, rerank: bool
) -> list[QuestionResult]:
    results = []
    for question in questions:
        started = time.perf_counter()
        found = await service.search(question.question, MRR_DEPTH, mode=mode, rerank=rerank)
        latency = (time.perf_counter() - started) * 1000
        results.append(score_question(question, found.hits, latency))
    return results


def summarize(
    config: str,
    settings: dict[str, Any],
    indexing: dict[str, float],
    results: list[QuestionResult],
) -> RunResult:
    types = sorted({r.type for r in results if r.recall})
    out_of_scope = [r.top1_score for r in results if not r.recall and r.top1_score is not None]
    in_scope = [r.top1_score for r in results if r.recall and r.top1_score is not None]
    latencies = [r.latency_ms for r in results]
    metrics = aggregate(results)
    if in_scope:
        metrics["top1_score_in_scope"] = round(statistics.fmean(in_scope), 4)
    if out_of_scope:
        metrics["top1_score_out_of_scope"] = round(statistics.fmean(out_of_scope), 4)
    return RunResult(
        config=config,
        date=datetime.date.today().isoformat(),
        settings=settings,
        indexing=indexing,
        metrics=metrics,
        by_type={t: aggregate([r for r in results if r.type == t]) for t in types},
        latency_ms={"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        questions=results,
    )


def print_table(runs: list[RunResult]) -> None:
    columns = ["recall@1", "recall@3", "recall@5"]
    header = (
        f"{'config':32} "
        + " ".join(f"{c:>9}" for c in columns)
        + f" {'MRR':>7} {'p50 мс':>8} {'p95 мс':>8}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        m = run.metrics
        print(
            f"{run.config:32} {m['recall@1']:9.3f} {m['recall@3']:9.3f} {m['recall@5']:9.3f} "
            f"{m['mrr']:7.3f} {run.latency_ms['p50']:8.1f} {run.latency_ms['p95']:8.1f}"
        )


def save(run: RunResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run.date}_{run.config}.json"
    path.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


async def evaluate(args: argparse.Namespace, database_url: str) -> list[RunResult]:
    overrides: dict[str, Any] = {
        "database_url": database_url,
        "qdrant_url": "http://localhost:6333",
        "admin_api_key": "eval",
        "embedder": args.embedder,
        "embedding_cache_size": 0,
        "reranker": "bge" if args.rerank else "none",
    }
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    settings = Settings(**overrides)
    embedder = await asyncio.to_thread(build_embedder, settings)
    reranker = await asyncio.to_thread(build_reranker, settings)
    database = Database(database_url)
    qdrant = Qdrant(location=":memory:")
    qdrant_store = QdrantVectorStore(qdrant.client, f"eval_{embedder.name}", embedder.dim)
    stores: dict[StoreName, VectorStore] = {
        "qdrant": qdrant_store,
        "pgvector": PgVectorStore(database),
    }
    chunker = Chunker(embedder.count_tokens, settings.chunk_size, settings.chunk_overlap)
    try:
        indexing = await index_corpus(args.data_dir, database, embedder, chunker, qdrant_store)
        service = SearchService(
            embedder,
            stores,
            SearchDefaults(store=args.store, rerank_candidates=settings.rerank_candidates),
            PostgresBM25(database),
            reranker,
        )
        questions = load_questions(QUESTIONS)
        runs = []
        for mode in args.modes:
            suffix = "_rerank" if args.rerank else ""
            config = (
                args.config or f"{mode}{suffix}_{embedder.name}_{args.store}_c{settings.chunk_size}"
            )
            results = await run_mode(service, questions, mode, args.rerank)
            described = {
                "mode": mode,
                "rerank": settings.reranker_model if args.rerank else None,
                "embedder": embedder.name,
                "store": args.store,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "rrf_k": settings.rrf_k,
                "questions_file": str(QUESTIONS.relative_to(API_DIR)),
            }
            runs.append(summarize(config, described, indexing, results))
        return runs
    finally:
        await database.close()
        await qdrant.close()
        await embedder.close()
        if reranker is not None:
            await reranker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="recall@1/3/5 и MRR на eval/questions.jsonl")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="папка корпуса (PDF, Markdown)"
    )
    parser.add_argument("--config", help="имя конфигурации для файла результата")
    parser.add_argument(
        "--mode",
        dest="modes",
        default="vector",
        type=lambda value: [m for m in value.split(",") if m],
        help="vector, bm25, hybrid или список через запятую",
    )
    parser.add_argument(
        "--embedder", default="hashing", choices=["hashing", "openai", "e5", "bge-m3"]
    )
    parser.add_argument("--store", default="qdrant", choices=["qdrant", "pgvector"])
    parser.add_argument(
        "--rerank", action="store_true", help="включить bge-reranker (локальная модель)"
    )
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    )
    args = parser.parse_args()
    unknown = [m for m in args.modes if m not in get_args(SearchMode)]
    if unknown:
        parser.error(f"неизвестный режим: {unknown}")
    if not args.data_dir.is_dir():
        parser.error(f"нет папки корпуса: {args.data_dir}")
    if args.config and len(args.modes) > 1:
        parser.error("--config задаётся только для одного режима")
    return args


def main() -> None:
    args = parse_args()
    database_url = eval_database_url(args.database_url)
    asyncio.run(ensure_database(args.database_url))
    migrate(database_url)
    runs = asyncio.run(evaluate(args, database_url))
    print_table(runs)
    for run in runs:
        print(f"сохранено: {save(run).relative_to(API_DIR)}")


if __name__ == "__main__":
    main()
