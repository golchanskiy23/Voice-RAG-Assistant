import uuid
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from app.adapters.vector_stores.base import SearchHit
from app.core.database import Database

BM25_K1 = 1.2
BM25_B = 0.75

BM25_QUERY = text(
    """
    WITH terms AS (
        SELECT DISTINCT lexeme FROM unnest(to_tsvector('russian', :query))
    ),
    q AS (
        SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS query FROM terms
    ),
    stats AS (
        SELECT count(*) AS n, greatest(avg(length(c.tsv)), 1) AS avgdl
        FROM chunks c
        WHERE CAST(:document_ids AS uuid[]) IS NULL
           OR c.document_id = ANY(CAST(:document_ids AS uuid[]))
    ),
    candidates AS (
        SELECT c.id, c.tsv
        FROM chunks c, q
        WHERE c.tsv @@ q.query
          AND (CAST(:document_ids AS uuid[]) IS NULL
               OR c.document_id = ANY(CAST(:document_ids AS uuid[])))
    ),
    matches AS (
        SELECT c.id, u.lexeme, coalesce(array_length(u.positions, 1), 1) AS tf,
               length(c.tsv) AS dl
        FROM candidates c, unnest(c.tsv) u
        WHERE u.lexeme IN (SELECT lexeme FROM terms)
    ),
    df AS (
        SELECT lexeme, count(*) AS df FROM matches GROUP BY lexeme
    ),
    ranked AS (
        SELECT m.id,
               sum(
                   ln(1 + (st.n - df.df + 0.5) / (df.df + 0.5))
                   * m.tf * (:k1 + 1)
                   / (m.tf + :k1 * (1 - :b + :b * m.dl / st.avgdl))
               ) AS score
        FROM matches m JOIN df USING (lexeme) CROSS JOIN stats st
        GROUP BY m.id
        ORDER BY score DESC
        LIMIT :k
    )
    SELECT c.id, c.document_id, d.filename, c.ord, c.text, c.page, c.section, r.score
    FROM ranked r
    JOIN chunks c ON c.id = r.id
    JOIN documents d ON d.id = c.document_id
    ORDER BY r.score DESC
    """
).bindparams(bindparam("document_ids", type_=ARRAY(UUID(as_uuid=True))))


class KeywordIndex(Protocol):
    async def search(
        self, query: str, k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]: ...


class PostgresBM25:
    def __init__(self, database: Database, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._database = database
        self._k1 = k1
        self._b = b

    async def search(
        self, query: str, k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        params = {
            "query": query,
            "k": k,
            "k1": self._k1,
            "b": self._b,
            "document_ids": list(document_ids) if document_ids else None,
        }
        async with self._database.sessionmaker() as session:
            rows = (await session.execute(BM25_QUERY, params)).all()
        return [
            SearchHit(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                ord=row.ord,
                text=row.text,
                page=row.page,
                section=row.section,
                score=float(row.score),
                found_by=("bm25",),
            )
            for row in rows
        ]
