import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.integration.conftest import run_alembic

pytestmark = pytest.mark.integration

EXPECTED_INDEXES = {
    "documents_pkey",
    "documents_sha256_key",
    "chunks_pkey",
    "chunks_document_id_ord_key",
    "chunks_embedding_hnsw",
    "chunks_tsv_gin",
}


def fetch(url: str, *statements: str) -> list[Sequence[Any]]:
    async def run() -> list[Sequence[Any]]:
        engine = create_async_engine(url)
        try:
            async with engine.connect() as connection:
                result: list[Sequence[Any]] = []
                for statement in statements:
                    rows = await connection.execute(text(statement))
                    result.extend(rows.all() if rows.returns_rows else [])
                await connection.rollback()
                return result
        finally:
            await engine.dispose()

    return asyncio.run(run())


def table_names(url: str) -> set[str]:
    rows = fetch(url, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    return {row[0] for row in rows}


def test_upgrade_creates_schema(migrated_database_url: str) -> None:
    extensions = fetch(
        migrated_database_url, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    indexes = fetch(
        migrated_database_url, "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
    )
    version = fetch(migrated_database_url, "SELECT version_num FROM alembic_version")

    assert extensions == [("0.8.0",)]
    assert {"documents", "chunks"} <= table_names(migrated_database_url)
    assert {row[0] for row in indexes} >= EXPECTED_INDEXES
    assert version == [("0001",)]


def test_downgrade_then_upgrade(migrated_database_url: str) -> None:
    run_alembic(migrated_database_url, "downgrade", "base")
    assert not {"documents", "chunks"} & table_names(migrated_database_url)

    run_alembic(migrated_database_url, "upgrade", "head")
    assert {"documents", "chunks"} <= table_names(migrated_database_url)


def test_upgrade_is_idempotent(migrated_database_url: str) -> None:
    run_alembic(migrated_database_url, "upgrade", "head")

    assert fetch(migrated_database_url, "SELECT count(*) FROM alembic_version") == [(1,)]


INSERT_DOCUMENT = """
    INSERT INTO documents (filename, sha256, mime, pages)
    VALUES ('a.md', repeat('a', 64), 'text/markdown', 1)
"""
INSERT_CHUNK = """
    INSERT INTO chunks (document_id, ord, text, tokens, embedding)
    SELECT id, 0, 'Векторный поиск находит документы', 4, array_fill(0.1, ARRAY[1024])::vector
    FROM documents
"""


def test_tsv_is_generated_with_russian_stemming(migrated_database_url: str) -> None:
    rows = fetch(
        migrated_database_url,
        INSERT_DOCUMENT,
        INSERT_CHUNK,
        "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('russian', 'документ')",
    )

    assert rows == [(1,)]


def test_deleting_document_cascades_to_chunks(migrated_database_url: str) -> None:
    rows = fetch(
        migrated_database_url,
        INSERT_DOCUMENT,
        INSERT_CHUNK,
        "DELETE FROM documents",
        "SELECT count(*) FROM chunks",
    )

    assert rows == [(0,)]


def test_cosine_distance_to_itself_is_zero(migrated_database_url: str) -> None:
    rows = fetch(
        migrated_database_url,
        INSERT_DOCUMENT,
        INSERT_CHUNK,
        "SELECT embedding <=> array_fill(0.1, ARRAY[1024])::vector FROM chunks",
    )

    assert rows == [(0.0,)]


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "INSERT INTO chunks (document_id, ord, text, tokens, embedding)"
            " VALUES (gen_random_uuid(), 0, 't', 1, '[1,2,3]')",
            "expected 1024 dimensions, not 3",
        ),
        (INSERT_DOCUMENT + ";" + INSERT_DOCUMENT, "documents_sha256_key"),
        (
            "INSERT INTO chunks (document_id, ord, text, tokens)"
            " VALUES (gen_random_uuid(), 0, 't', 1)",
            "chunks_document_id_fkey",
        ),
    ],
)
def test_constraints_reject_invalid_rows(
    migrated_database_url: str, statement: str, message: str
) -> None:
    with pytest.raises(DBAPIError, match=message):
        fetch(migrated_database_url, *statement.split(";"))
