import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

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
class StoredDocument:
    id: uuid.UUID
    filename: str
    sha256: str
    mime: str
    pages: int | None
    chunks: int


class DocumentRepository(Protocol):
    async def find_by_sha256(self, sha256: str) -> StoredDocument | None: ...

    async def save(
        self, document: NewDocument, chunks: Sequence[NewChunk]
    ) -> tuple[StoredDocument, SaveStatus]: ...


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_sha256(self, sha256: str) -> StoredDocument | None:
        chunk_count = (
            select(func.count(Chunk.id)).where(Chunk.document_id == Document.id).scalar_subquery()
        )
        row = (
            await self._session.execute(
                select(Document, chunk_count).where(Document.sha256 == sha256)
            )
        ).one_or_none()
        if row is None:
            return None
        document, chunks = row
        return StoredDocument(
            id=document.id,
            filename=document.filename,
            sha256=document.sha256,
            mime=document.mime,
            pages=document.pages,
            chunks=chunks,
        )

    async def save(
        self, document: NewDocument, chunks: Sequence[NewChunk]
    ) -> tuple[StoredDocument, SaveStatus]:
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
            return existing, "exists"

        replaced = await self._session.scalars(
            delete(Document)
            .where(Document.filename == document.filename, Document.id != inserted)
            .returning(Document.id)
        )
        status: SaveStatus = "replaced" if replaced.first() is not None else "created"
        if chunks:
            await self._session.execute(
                insert(Chunk),
                [
                    {
                        "document_id": inserted,
                        "ord": chunk.ord,
                        "text": chunk.text,
                        "page": chunk.page,
                        "section": chunk.section,
                        "tokens": chunk.tokens,
                        "embedding": chunk.embedding,
                        "meta": {"pages": list(chunk.pages)},
                    }
                    for chunk in chunks
                ],
            )
        await self._session.commit()
        stored = StoredDocument(
            id=inserted,
            filename=document.filename,
            sha256=document.sha256,
            mime=document.mime,
            pages=document.pages,
            chunks=len(chunks),
        )
        return stored, status
