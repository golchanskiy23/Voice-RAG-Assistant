import argparse
import asyncio
import random
import statistics
import time
import uuid
from collections.abc import Awaitable, Callable

from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete, insert, text

from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.vector_stores.base import VectorPoint, VectorStore
from app.adapters.vector_stores.pgvector import PgVectorStore
from app.adapters.vector_stores.qdrant import QdrantVectorStore
from app.core.database import Database
from app.repositories.models import Chunk, Document

COLLECTION = "chunks_bench"
CHUNKS_PER_DOCUMENT = 100
PG_BATCH = 1000
GREEN_POLLS = 3000
SYLLABLES = [
    "ра",
    "до",
    "ко",
    "ми",
    "ста",
    "ли",
    "ны",
    "ве",
    "то",
    "сы",
    "пре",
    "об",
    "за",
    "ни",
    "ка",
]


def make_vocabulary(rng: random.Random, size: int) -> list[str]:
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def make_texts(rng: random.Random, vocabulary: list[str], count: int) -> list[str]:
    weights = [1 / (rank + 1) for rank in range(len(vocabulary))]
    return [" ".join(rng.choices(vocabulary, weights, k=rng.randint(40, 80))) for _ in range(count)]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


async def timed(operation: Callable[[], Awaitable[object]]) -> float:
    start = time.perf_counter()
    await operation()
    return (time.perf_counter() - start) * 1000


async def latency(
    store: VectorStore, queries: list[list[float]], document_ids: list[uuid.UUID] | None
) -> tuple[float, float]:
    for vector in queries[:10]:
        await store.search(vector, 5, document_ids)
    samples: list[float] = []
    for vector in queries:
        start = time.perf_counter()
        await store.search(vector, 5, document_ids)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), percentile(samples, 95)


async def exact_top5(database: Database, vector: list[float]) -> list[uuid.UUID]:
    async with database.engine.connect() as connection:
        await connection.execute(text("SET LOCAL enable_indexscan = off"))
        rows = await connection.execute(
            text(
                "SELECT id FROM chunks WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 5"
            ),
            {"v": str(vector)},
        )
        return [row[0] for row in rows]


async def main(database_url: str, qdrant_url: str, chunks: int, queries: int) -> None:
    rng = random.Random(42)
    embedder = HashingEmbedder()
    vocabulary = make_vocabulary(rng, 5000)
    texts = make_texts(rng, vocabulary, chunks)
    query_vectors = await embedder.embed_documents(make_texts(rng, vocabulary, queries))
    vectors = await embedder.embed_documents(texts)
    document_ids = [uuid.uuid4() for _ in range(0, chunks, CHUNKS_PER_DOCUMENT)]
    points = [
        VectorPoint(
            chunk_id=uuid.uuid4(),
            document_id=document_ids[i // CHUNKS_PER_DOCUMENT],
            filename=f"bench-{i // CHUNKS_PER_DOCUMENT}.md",
            ord=i % CHUNKS_PER_DOCUMENT,
            text=texts[i],
            page=None,
            section="bench",
            vector=vectors[i],
        )
        for i in range(chunks)
    ]

    database = Database(database_url)
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False, timeout=120)
    pg = PgVectorStore(database)
    qdrant = QdrantVectorStore(client, COLLECTION, embedder.dim)
    try:
        async with database.engine.begin() as connection:
            await connection.execute(delete(Document).where(Document.filename.like("bench-%")))
        if await client.collection_exists(COLLECTION):
            await client.delete_collection(COLLECTION)

        async def index_pg() -> None:
            async with database.engine.begin() as connection:
                await connection.execute(
                    insert(Document),
                    [
                        {
                            "id": d,
                            "filename": f"bench-{n}.md",
                            "sha256": uuid.uuid4().hex * 2,
                            "mime": "text/markdown",
                        }
                        for n, d in enumerate(document_ids)
                    ],
                )
            for start in range(0, chunks, PG_BATCH):
                async with database.engine.begin() as connection:
                    await connection.execute(
                        insert(Chunk),
                        [
                            {
                                "id": p.chunk_id,
                                "document_id": p.document_id,
                                "ord": p.ord,
                                "text": p.text,
                                "section": p.section,
                                "tokens": embedder.count_tokens(p.text),
                                "embedding": p.vector,
                            }
                            for p in points[start : start + PG_BATCH]
                        ],
                    )

        async def index_qdrant() -> None:
            await qdrant.upsert(points)

        async def qdrant_green() -> None:
            for _ in range(GREEN_POLLS):
                if (await client.get_collection(COLLECTION)).status == "green":
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError("коллекция Qdrant не перешла в green")

        pg_index_ms = await timed(index_pg)
        qdrant_upsert_ms = await timed(index_qdrant)
        qdrant_green_ms = await timed(qdrant_green)

        exact = [await exact_top5(database, v) for v in query_vectors]
        report = {}
        for store in (pg, qdrant):
            found = [[h.chunk_id for h in await store.search(v, 5)] for v in query_vectors]
            recall = statistics.mean(
                len(set(f) & set(e)) / 5 for f, e in zip(found, exact, strict=True)
            )
            p50, p95 = await latency(store, query_vectors, None)
            f50, f95 = await latency(store, query_vectors, [document_ids[0]])
            report[store.name] = (recall, p50, p95, f50, f95, found)

        overlap = statistics.mean(
            len(set(a) & set(b)) / 5
            for a, b in zip(report["pgvector"][5], report["qdrant"][5], strict=True)
        )

        print(f"chunks={chunks}, dim={embedder.dim}, queries={queries}, k=5, ef_search=64")
        print(f"индексация: pgvector {pg_index_ms / 1000:.1f} с (INSERT + построение HNSW)")
        print(
            f"индексация: qdrant upsert {qdrant_upsert_ms / 1000:.1f} с, "
            f"+ оптимизация до green {qdrant_green_ms / 1000:.1f} с"
        )
        print(
            "| Хранилище | recall@5 | p50, мс | p95, мс | p50 с фильтром, мс | p95 с фильтром, мс |"
        )
        print("|---|---|---|---|---|---|")
        for name, (recall, p50, p95, f50, f95, _) in report.items():
            print(f"| {name} | {recall:.3f} | {p50:.1f} | {p95:.1f} | {f50:.1f} | {f95:.1f} |")
        print(f"совпадение top-5 pgvector и qdrant: {overlap:.3f}")
    finally:
        async with database.engine.begin() as connection:
            await connection.execute(delete(Document).where(Document.filename.like("bench-%")))
        if await client.collection_exists(COLLECTION):
            await client.delete_collection(COLLECTION)
        await client.close()
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="postgresql+asyncpg://rag:rag@localhost:5433/rag")
    parser.add_argument("--qdrant-url", default="http://localhost:6334")
    parser.add_argument("--chunks", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.database_url, args.qdrant_url, args.chunks, args.queries))
