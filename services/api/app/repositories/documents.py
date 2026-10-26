import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import ScalarSelect

from app.repositories.models import Chunk, Document

SaveStatus = Literal["created", "replaced", "exists"]


@dataclass(frozen=True)
class NewDocument:
    filename: str
    sha256: str
    mime: str
    pages: int | None = None


@dataclass(frozen=True)
class NewChunk:
    ord: int
    text: str
    tokens: int
    embedding: list[float]
    section: str | None
    pages: tuple[int, ...]

    @property
    def page(self) -> int | None:
        return self.pages[0] if self.pages else None


@dataclass(frozen=True)
class SavedChunk:
    id: uuid.UUID
    chunk: NewChunk


@dataclass(frozen=True)
class StoredDocument:
    id: uuid.UUID
    filename: str
    sha256: str
    mime: str
    pages: int | None
    chunks: int


@dataclass(frozen=True)
class SaveResult:
    document: StoredDocument
    status: SaveStatus
    chunks: list[SavedChunk]
    replaced: list[uuid.UUID]


class DocumentRepository(Protocol):
    async def find_by_sha256(self, sha256: str) -> StoredDocument | None: ...

    async def save(self, document: NewDocument, chunks: Sequence[NewChunk]) -> SaveResult: ...

    async def list_all(self) -> list[StoredDocument]: ...

    async def delete(self, document_id: uuid.UUID) -> bool: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


def chunk_count() -> ScalarSelect[int]:
    return select(func.count(Chunk.id)).where(Chunk.document_id == Document.id).scalar_subquery()


def to_stored(document: Document, chunks: int) -> StoredDocument:
    return StoredDocument(
        id=document.id,
        filename=document.filename,
        sha256=document.sha256,
        mime=document.mime,
        pages=document.pages,
        chunks=chunks,
    )


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_sha256(self, sha256: str) -> StoredDocument | None:
        row = (
            await self._session.execute(
                select(Document, chunk_count()).where(Document.sha256 == sha256)
            )
        ).one_or_none()
        return None if row is None else to_stored(*row)

    async def list_all(self) -> list[StoredDocument]:
        rows = await self._session.execute(
            select(Document, chunk_count()).order_by(Document.created_at, Document.filename)
        )
        return [to_stored(document, chunks) for document, chunks in rows]

    async def delete(self, document_id: uuid.UUID) -> bool:
        deleted = await self._session.scalar(
            delete(Document).where(Document.id == document_id).returning(Document.id)
        )
        return deleted is not None

    async def save(self, document: NewDocument, chunks: Sequence[NewChunk]) -> SaveResult:
        inserted = await self._session.scalar(
            pg_insert(Document)
            .values(
                filename=document.filename,
                sha256=document.sha256,
                mime=document.mime,
                pages=document.pages,
            )
            .on_conflict_do_nothing(index_elements=[Document.sha256])
            .returning(Document.id)
        )
        if inserted is None:
            await self._session.rollback()
            existing = await self.find_by_sha256(document.sha256)
            if existing is None:
                raise LookupError(f"документ {document.sha256} не найден после конфликта вставки")
            return SaveResult(existing, "exists", [], [])

        replaced = list(
            await self._session.scalars(
                delete(Document)
                .where(Document.filename == document.filename, Document.id != inserted)
                .returning(Document.id)
            )
        )
        saved = [SavedChunk(id=uuid.uuid4(), chunk=chunk) for chunk in chunks]
        if saved:
            await self._session.execute(
                insert(Chunk),
                [
                    {
                        "id": item.id,
                        "document_id": inserted,
                        "ord": item.chunk.ord,
                        "text": item.chunk.text,
                        "page": item.chunk.page,
                        "section": item.chunk.section,
                        "tokens": item.chunk.tokens,
                        "embedding": item.chunk.embedding,
                        "meta": {"pages": list(item.chunk.pages)},
                    }
                    for item in saved
                ],
            )
        stored = StoredDocument(
            id=inserted,
            filename=document.filename,
            sha256=document.sha256,
            mime=document.mime,
            pages=document.pages,
            chunks=len(chunks),
        )
        return SaveResult(stored, "replaced" if replaced else "created", saved, replaced)

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
