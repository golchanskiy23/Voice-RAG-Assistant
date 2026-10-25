import asyncio
import hashlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.security import API_KEY_HEADER
from app.core.config import Settings
from app.main import create_app
from app.repositories.documents import NewDocument, SqlDocumentRepository
from app.repositories.models import Base, Document
from tests.factories import TEST_ADMIN_API_KEY

pytestmark = pytest.mark.integration

MARKDOWN = "# Интеграция\n\nТекст.".encode()


@pytest.fixture(autouse=True)
def clean_documents(migrated_database_url: str) -> Iterator[None]:
    yield

    async def clean() -> None:
        engine = create_async_engine(migrated_database_url)
        async with engine.begin() as connection:
            await connection.execute(delete(Document))
        await engine.dispose()

    asyncio.run(clean())


def table_of(diff: Any) -> str | None:
    entry = diff[0] if isinstance(diff, list) else diff
    target = entry[1]
    if hasattr(target, "table"):
        return str(target.table.name)
    if hasattr(target, "name") and entry[0].endswith("_table"):
        return str(target.name)
    return str(entry[2]) if len(entry) > 2 and isinstance(entry[2], str) else None


def test_model_matches_migrated_schema(migrated_database_url: str) -> None:
    def compare(connection: Connection) -> list[Any]:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        return list(compare_metadata(context, Base.metadata))

    async def run() -> list[Any]:
        engine = create_async_engine(migrated_database_url)
        async with engine.connect() as connection:
            diffs = await connection.run_sync(compare)
        await engine.dispose()
        return diffs

    diffs = [d for d in asyncio.run(run()) if table_of(d) in Base.metadata.tables]

    assert diffs == []


def test_upload_persists_document(settings: Settings, migrated_database_url: str) -> None:
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/documents",
            files={"file": ("guide.md", MARKDOWN)},
            headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
        )
        duplicate = client.post(
            "/documents",
            files={"file": ("copy.md", MARKDOWN)},
            headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
        )

    async def load() -> list[Document]:
        engine = create_async_engine(migrated_database_url)
        async with async_sessionmaker(engine)() as session:
            rows = list(await session.scalars(select(Document)))
        await engine.dispose()
        return rows

    rows = asyncio.run(load())
    assert response.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == response.json()["id"]
    assert len(rows) == 1
    assert rows[0].id == uuid.UUID(response.json()["id"])
    assert (rows[0].filename, rows[0].mime) == ("guide.md", "text/markdown")
    assert rows[0].sha256 == hashlib.sha256(MARKDOWN).hexdigest()
    assert rows[0].created_at.tzinfo is not None


def test_concurrent_uploads_of_same_file_create_one_document(migrated_database_url: str) -> None:
    document = NewDocument(filename="race.md", sha256="b" * 64, mime="text/markdown")

    async def run() -> list[bool]:
        engine = create_async_engine(migrated_database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        async def attempt() -> bool:
            session: AsyncSession
            async with sessions() as session:
                _, created = await SqlDocumentRepository(session).get_or_create(document)
                return created

        results = await asyncio.gather(*(attempt() for _ in range(10)))
        await engine.dispose()
        return list(results)

    results = asyncio.run(run())

    assert results.count(True) == 1
    assert results.count(False) == 9
