import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.adapters.vector_stores.base import SearchHit, StoreName, VectorPoint

UPSERT_BATCH = 256


def document_filter(document_ids: Sequence[uuid.UUID]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id", match=models.MatchAny(any=[str(d) for d in document_ids])
            )
        ]
    )


class QdrantVectorStore:
    def __init__(
        self, client: AsyncQdrantClient, collection: str, dim: int, ef_search: int = 64
    ) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim
        self._ef_search = ef_search
        self._ensured = False

    @property
    def name(self) -> StoreName:
        return "qdrant"

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self) -> None:
        if self._ensured:
            return
        if await self._client.collection_exists(self._collection):
            self._ensured = True
            return
        await self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64),
        )
        await self._client.create_payload_index(
            self._collection, "document_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        self._ensured = True

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        await self.ensure_collection()
        for start in range(0, len(points), UPSERT_BATCH):
            batch = points[start : start + UPSERT_BATCH]
            await self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(id=str(p.chunk_id), vector=p.vector, payload=payload(p))
                    for p in batch
                ],
                wait=True,
            )

    async def search(
        self, vector: list[float], k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        await self.ensure_collection()
        response = await self._client.query_points(
            self._collection,
            query=vector,
            limit=k,
            query_filter=document_filter(document_ids) if document_ids else None,
            search_params=models.SearchParams(hnsw_ef=self._ef_search),
            with_payload=True,
        )
        return [to_hit(point.id, point.score, point.payload or {}) for point in response.points]

    async def delete_documents(self, document_ids: Sequence[uuid.UUID]) -> None:
        if not document_ids:
            return
        await self.ensure_collection()
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=document_filter(document_ids)),
            wait=True,
        )


def payload(point: VectorPoint) -> dict[str, Any]:
    return {
        "document_id": str(point.document_id),
        "filename": point.filename,
        "ord": point.ord,
        "text": point.text,
        "page": point.page,
        "section": point.section,
    }


def to_hit(point_id: Any, score: float, data: dict[str, Any]) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(str(point_id)),
        document_id=uuid.UUID(data["document_id"]),
        filename=data["filename"],
        ord=data["ord"],
        text=data["text"],
        page=data.get("page"),
        section=data.get("section"),
        score=float(score),
    )
