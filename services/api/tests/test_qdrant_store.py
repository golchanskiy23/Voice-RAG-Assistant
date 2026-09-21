import asyncio
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from app.adapters.vector_stores.base import SearchHit, VectorPoint
from app.adapters.vector_stores.qdrant import QdrantVectorStore

FIRST, SECOND = uuid.uuid4(), uuid.uuid4()
QUERY = [0.9, 0.1, 0.0]


def point(document_id: uuid.UUID, ord: int, vector: list[float]) -> VectorPoint:
    return VectorPoint(
        chunk_id=uuid.uuid4(),
        document_id=document_id,
        filename=f"{document_id}.md",
        ord=ord,
        text=f"чанк {ord}",
        page=ord + 1,
        section="Раздел",
        vector=vector,
    )


POINTS = [
    point(FIRST, 0, [1.0, 0.0, 0.0]),
    point(FIRST, 1, [0.7, 0.7, 0.0]),
    point(SECOND, 0, [0.0, 1.0, 0.0]),
    point(SECOND, 1, [0.0, 0.0, 1.0]),
]


@dataclass
class Outcome:
    top: list[SearchHit]
    filtered: list[SearchHit]
    after_delete: list[SearchHit]
    count: int


async def scenario() -> Outcome:
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client, "chunks_test", dim=3)
    await store.ensure_collection()
    await store.ensure_collection()
    await store.upsert(POINTS)
    await store.upsert(POINTS[:1])
    count = (await client.count("chunks_test")).count
    top = await store.search(QUERY, k=2)
    filtered = await store.search(QUERY, k=5, document_ids=[SECOND])
    await store.delete_documents([FIRST])
    after_delete = await store.search(QUERY, k=5)
    return Outcome(top, filtered, after_delete, count)


def test_upsert_is_idempotent_by_chunk_id() -> None:
    assert asyncio.run(scenario()).count == len(POINTS)


def test_search_returns_nearest_with_payload() -> None:
    top = asyncio.run(scenario()).top

    assert [h.chunk_id for h in top] == [POINTS[0].chunk_id, POINTS[1].chunk_id]
    assert top[0].score > top[1].score
    assert (top[0].filename, top[0].page, top[0].section, top[0].text) == (
        POINTS[0].filename,
        1,
        "Раздел",
        "чанк 0",
    )


def test_filter_by_document_and_delete() -> None:
    outcome = asyncio.run(scenario())

    assert {h.document_id for h in outcome.filtered} == {SECOND}
    assert {h.document_id for h in outcome.after_delete} == {SECOND}
