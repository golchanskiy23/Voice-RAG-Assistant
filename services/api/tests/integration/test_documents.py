import asyncio
import hashlib
from pathlib import Path
from typing import Any

import httpx2
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.embedders.base import EMBEDDING_DIM
from app.api.security import API_KEY_HEADER
from app.core.config import Settings
from app.main import create_app
from app.repositories.documents import NewChunk, NewDocument, SqlDocumentRepository
from app.repositories.models import Base, Chunk, Document
from tests.factories import TEST_ADMIN_API_KEY

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PDF = (FIXTURES / "regulation_ru.pdf").read_bytes()
MARKDOWN = (
    "# Руководство\n\n## Загрузка\n\n"
    "Документы загружаются через API. Ключ передаётся в заголовке.\n"
).encode()


@pytest.fixture(autouse=True)
def clean(clean_stores: None) -> None:
    return None


def query(url: str, statement: Any) -> list[Any]:
    async def run() -> list[Any]:
        engine = create_async_engine(url)
        async with async_sessionmaker(engine)() as session:
            rows = list((await session.execute(statement)).all())
        await engine.dispose()
        return rows

    return asyncio.run(run())


def upload(client: TestClient, filename: str, content: bytes) -> httpx2.Response:
    response: httpx2.Response = client.post(
        "/documents",
        files={"file": (filename, content)},
        headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
    )
    return response


def table_of(diff: Any) -> str | None:
    entry = diff[0] if isinstance(diff, list) else diff
    target = entry[1]
    if hasattr(target, "table"):
        return str(target.table.name)
    if hasattr(target, "name") and entry[0].endswith("_table"):
        return str(target.name)
    return str(entry[2]) if len(entry) > 2 and isinstance(entry[2], str) else None


def test_models_match_migrated_schema(migrated_database_url: str) -> None:
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


def test_pdf_upload_writes_chunks(settings: Settings, migrated_database_url: str) -> None:
    with TestClient(create_app(settings=settings)) as client:
        response = upload(client, "regulation.pdf", PDF)

    body = response.json()
    rows = query(
        migrated_database_url,
        select(
            Chunk.ord,
            Chunk.page,
            Chunk.section,
            Chunk.tokens,
            func.vector_dims(Chunk.embedding),
            Chunk.tsv.is_not(None),
            Chunk.meta,
        ).order_by(Chunk.ord),
    )
    documents = query(migrated_database_url, select(Document.pages, Document.sha256))

    assert response.status_code == 201
    assert body["chunks"] == len(rows) > 0
    assert documents == [(2, hashlib.sha256(PDF).hexdigest())]
    assert [r[0] for r in rows] == list(range(len(rows)))
    assert {r[1] for r in rows} <= {1, 2}
    assert all(r[2].startswith("Регламент обработки документов > ") for r in rows)
    assert all(0 < r[3] <= settings.chunk_size for r in rows)
    assert all(r[4] == EMBEDDING_DIM for r in rows)
    assert all(r[5] for r in rows)
    assert any(r[6]["pages"] == [1, 2] for r in rows)


def test_full_text_search_finds_uploaded_chunk(
    settings: Settings, migrated_database_url: str
) -> None:
    with TestClient(create_app(settings=settings)) as client:
        upload(client, "guide.md", MARKDOWN)

    rows = query(
        migrated_database_url,
        select(Chunk.section).where(
            Chunk.tsv.op("@@")(func.plainto_tsquery("russian", "загружать документ"))
        ),
    )

    assert rows == [("Руководство > Загрузка",)]


def test_repeated_upload_does_not_duplicate_chunks(
    settings: Settings, migrated_database_url: str
) -> None:
    with TestClient(create_app(settings=settings)) as client:
        first = upload(client, "guide.md", MARKDOWN)
        second = upload(client, "copy.md", MARKDOWN)

    counts = query(
        migrated_database_url,
        select(func.count(Document.id), func.count(Chunk.id)).select_from(Document).join(Chunk),
    )

    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["status"] == "exists"
    assert second.json()["chunks"] == first.json()["chunks"]
    assert counts == [(first.json()["chunks"], first.json()["chunks"])]


def test_new_version_replaces_old_chunks(settings: Settings, migrated_database_url: str) -> None:
    updated = (
        "# Руководство\n\n## Загрузка\n\nНовая редакция: загрузка только через веб-интерфейс.\n"
    )
    with TestClient(create_app(settings=settings)) as client:
        first = upload(client, "guide.md", MARKDOWN)
        second = upload(client, "guide.md", updated.encode())

    documents = query(migrated_database_url, select(Document.id))
    texts = [r[0] for r in query(migrated_database_url, select(Chunk.text))]

    assert second.json()["status"] == "replaced"
    assert [str(r[0]) for r in documents] == [second.json()["id"]]
    assert first.json()["id"] != second.json()["id"]
    assert texts == ["Новая редакция: загрузка только через веб-интерфейс."]


def test_concurrent_saves_of_same_file_create_one_document(migrated_database_url: str) -> None:
    document = NewDocument(filename="race.md", sha256="b" * 64, mime="text/markdown")
    chunk = NewChunk(
        ord=0, text="текст", tokens=1, embedding=[0.0] * EMBEDDING_DIM, section=None, pages=()
    )

    async def run() -> list[str]:
        engine = create_async_engine(migrated_database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)

        async def attempt() -> str:
            session: AsyncSession
            async with sessions() as session:
                repository = SqlDocumentRepository(session)
                saved = await repository.save(document, [chunk])
                if saved.status != "exists":
                    await repository.commit()
                return saved.status

        statuses = await asyncio.gather(*(attempt() for _ in range(10)))
        await engine.dispose()
        return list(statuses)

    statuses = asyncio.run(run())
    chunk_count = query(migrated_database_url, select(func.count()).select_from(text("chunks")))

    assert sorted(statuses) == ["created"] + ["exists"] * 9
    assert chunk_count == [(1,)]


def test_list_and_delete_remove_document_from_both_stores(
    settings: Settings, migrated_database_url: str
) -> None:
    auth = {API_KEY_HEADER: TEST_ADMIN_API_KEY}
    with TestClient(create_app(settings=settings)) as client:
        kept = upload(client, "kept.md", MARKDOWN).json()
        removed = upload(client, "removed.pdf", PDF).json()
        listed = client.get("/documents", headers=auth).json()["documents"]
        deleted = client.delete(f"/documents/{removed['id']}", headers=auth)
        search = client.post("/search", json={"query": "регламент документов", "k": 10})
        missing = client.delete(f"/documents/{removed['id']}", headers=auth)

    assert [d["filename"] for d in listed] == ["kept.md", "removed.pdf"]
    assert listed[1]["pages"] == removed["pages"]
    assert (deleted.status_code, missing.status_code) == (204, 404)
    assert {c["document_id"] for c in search.json()["chunks"]} <= {kept["id"]}
    chunk_documents = query(migrated_database_url, select(Chunk.document_id).distinct())
    assert [str(r[0]) for r in chunk_documents] == [kept["id"]]
