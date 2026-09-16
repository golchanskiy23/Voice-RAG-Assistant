import uuid
from collections.abc import Sequence

from sqlalchemy import bindparam, select, text, update

from app.adapters.vector_stores.base import SearchHit, StoreName, VectorPoint
from app.core.database import Database
from app.repositories.models import Chunk, Document


class PgVectorStore:
    def __init__(self, database: Database, ef_search: int = 64) -> None:
        self._database = database
        self._ef_search = ef_search

    @property
    def name(self) -> StoreName:
        return "pgvector"

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        statement = (
            update(Chunk)
            .where(Chunk.id == bindparam("chunk_id"))
            .values(embedding=bindparam("vector"))
            .execution_options(synchronize_session=False)
        )
        async with self._database.sessionmaker() as session:
            await session.execute(
                statement, [{"chunk_id": p.chunk_id, "vector": p.vector} for p in points]
            )
            await session.commit()

    async def search(
        self, vector: list[float], k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        rows = await self._query(vector, k, document_ids, exact=False)
        if len(rows) < k:
            rows = await self._query(vector, k, document_ids, exact=True)
        return [
            SearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                ord=chunk.ord,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                score=1.0 - float(row_distance),
            )
            for chunk, filename, row_distance in rows
        ]

    async def _query(
        self,
        vector: list[float],
        k: int,
        document_ids: Sequence[uuid.UUID] | None,
        exact: bool,
    ) -> list[tuple[Chunk, str, float]]:
        distance = Chunk.embedding.cosine_distance(vector).label("distance")
        query = (
            select(Chunk, Document.filename, distance)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        if document_ids:
            query = query.where(Chunk.document_id.in_(list(document_ids)))
        async with self._database.sessionmaker() as session:
            if exact:
                await session.execute(text("SET LOCAL enable_indexscan = off"))
                await session.execute(text("SET LOCAL enable_seqscan = on"))
            else:
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}"))
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            return list((await session.execute(query)).all())

    async def delete_documents(self, document_ids: Sequence[uuid.UUID]) -> None:
        if not document_ids:
            return
        async with self._database.sessionmaker() as session:
            await session.execute(
                update(Chunk)
                .where(Chunk.document_id.in_(list(document_ids)))
                .values(embedding=None)
            )
            await session.commit()
