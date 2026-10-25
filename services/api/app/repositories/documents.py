import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import Document


@dataclass(frozen=True)
class NewDocument:
    filename: str
    sha256: str
    mime: str


@dataclass(frozen=True)
class StoredDocument:
    id: uuid.UUID
    filename: str
    sha256: str
    mime: str


class DocumentRepository(Protocol):
    async def get_or_create(self, document: NewDocument) -> tuple[StoredDocument, bool]: ...


def to_stored(row: Document) -> StoredDocument:
    return StoredDocument(id=row.id, filename=row.filename, sha256=row.sha256, mime=row.mime)


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, document: NewDocument) -> tuple[StoredDocument, bool]:
        statement = (
            insert(Document)
            .values(filename=document.filename, sha256=document.sha256, mime=document.mime)
            .on_conflict_do_nothing(index_elements=[Document.sha256])
            .returning(Document)
        )
        created = (await self._session.scalars(statement)).one_or_none()
        if created is not None:
            await self._session.commit()
            return to_stored(created), True
        existing = await self._session.scalar(
            select(Document).where(Document.sha256 == document.sha256)
        )
        if existing is None:
            raise LookupError(f"документ {document.sha256} не найден после конфликта вставки")
        return to_stored(existing), False
