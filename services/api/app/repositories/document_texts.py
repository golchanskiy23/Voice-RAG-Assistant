import uuid
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from app.adapters.vector_stores.base import SearchHit
from app.core.database import Database

SIZES_QUERY = text(
    """
    SELECT d.id, coalesce(sum(length(c.text)), 0) AS chars
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    GROUP BY d.id, d.created_at
    ORDER BY d.created_at, d.id
    """
)

CHUNKS_QUERY = text(
    """
    SELECT c.id, c.document_id, d.filename, c.ord, c.text, c.page, c.section
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.document_id = ANY(CAST(:document_ids AS uuid[]))
    ORDER BY array_position(CAST(:document_ids AS uuid[]), c.document_id), c.ord
    """
).bindparams(bindparam("document_ids", type_=ARRAY(UUID(as_uuid=True))))


class DocumentTexts(Protocol):
    async def sizes(self) -> dict[uuid.UUID, int]: ...

    async def chunks(self, document_ids: Sequence[uuid.UUID]) -> list[SearchHit]: ...


class PostgresDocumentTexts:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def sizes(self) -> dict[uuid.UUID, int]:
        async with self._database.sessionmaker() as session:
            rows = (await session.execute(SIZES_QUERY)).all()
        return {row.id: int(row.chars) for row in rows}

    async def chunks(self, document_ids: Sequence[uuid.UUID]) -> list[SearchHit]:
        async with self._database.sessionmaker() as session:
            rows = (await session.execute(CHUNKS_QUERY, {"document_ids": list(document_ids)})).all()
        return [
            SearchHit(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                ord=row.ord,
                text=row.text,
                page=row.page,
                section=row.section,
                score=1.0,
                found_by=(),
            )
            for row in rows
        ]
