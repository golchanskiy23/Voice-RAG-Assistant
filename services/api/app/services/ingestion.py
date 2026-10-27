import asyncio
import contextlib
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from app.adapters.embedders.base import Embedder
from app.adapters.vector_stores.base import VectorPoint, VectorStore
from app.core.errors import AppError
from app.parsing.markdown import parse_markdown
from app.parsing.models import DocumentParseError, ParsedDocument
from app.parsing.pdf import parse_pdf
from app.repositories.documents import (
    DocumentRepository,
    NewChunk,
    NewDocument,
    SaveResult,
    SaveStatus,
    StoredDocument,
)
from app.services.chunking import Chunker

Mime = Literal["application/pdf", "text/markdown"]

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
EXTENSIONS: dict[str, Mime] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
PDF_SIGNATURE = b"%PDF-"


@dataclass(frozen=True)
class UploadResult:
    document: StoredDocument
    status: SaveStatus


def unsupported(message: str) -> AppError:
    return AppError("unsupported_media_type", message, status_code=415)


def detect_mime(filename: str, content: bytes) -> Mime:
    mime = EXTENSIONS.get(PurePath(filename).suffix.lower())
    if mime is None:
        raise unsupported("Поддерживаются только PDF и Markdown (.pdf, .md, .markdown)")
    if mime == "application/pdf" and not content.startswith(PDF_SIGNATURE):
        raise unsupported("Файл с расширением .pdf не является PDF")
    if mime == "text/markdown":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            raise unsupported("Markdown-файл должен быть в кодировке UTF-8") from None
    return mime


def parse(mime: Mime, content: bytes) -> ParsedDocument:
    return parse_pdf(content) if mime == "application/pdf" else parse_markdown(content)


def to_points(saved: SaveResult) -> list[VectorPoint]:
    return [
        VectorPoint(
            chunk_id=item.id,
            document_id=saved.document.id,
            filename=saved.document.filename,
            ord=item.chunk.ord,
            text=item.chunk.text,
            page=item.chunk.page,
            section=item.chunk.section,
            vector=item.chunk.embedding,
        )
        for item in saved.chunks
    ]


class IngestionService:
    def __init__(
        self,
        documents: DocumentRepository,
        embedder: Embedder,
        chunker: Chunker,
        index: VectorStore,
    ) -> None:
        self._documents = documents
        self._embedder = embedder
        self._chunker = chunker
        self._index = index

    async def upload(self, filename: str, content: bytes) -> UploadResult:
        if not content:
            raise AppError("empty_file", "Файл пустой", status_code=400)
        mime = detect_mime(filename, content)
        sha256 = hashlib.sha256(content).hexdigest()

        existing = await self._documents.find_by_sha256(sha256)
        if existing is not None:
            return UploadResult(existing, "exists")

        try:
            parsed = await asyncio.to_thread(parse, mime, content)
        except DocumentParseError as exc:
            raise AppError(exc.code, exc.message, status_code=422) from exc
        drafts = await asyncio.to_thread(self._chunker.split, parsed.blocks)
        vectors = await self._embedder.embed_documents([draft.text for draft in drafts])
        chunks = [
            NewChunk(
                ord=draft.ord,
                text=draft.text,
                tokens=draft.tokens,
                embedding=vector,
                section=draft.section,
                pages=draft.pages,
            )
            for draft, vector in zip(drafts, vectors, strict=True)
        ]
        document = NewDocument(
            filename=PurePath(filename).name, sha256=sha256, mime=mime, pages=parsed.pages
        )
        saved = await self._documents.save(document, chunks)
        if saved.status == "exists":
            return UploadResult(saved.document, "exists")
        try:
            await self._index.upsert(to_points(saved))
            await self._documents.commit()
        except Exception as exc:
            await self._documents.rollback()
            with contextlib.suppress(Exception):
                await self._index.delete_documents([saved.document.id])
            raise AppError(
                "indexing_failed", "Не удалось проиндексировать документ", status_code=503
            ) from exc
        with contextlib.suppress(Exception):
            await self._index.delete_documents(saved.replaced)
        return UploadResult(saved.document, saved.status)

    async def list_documents(self) -> list[StoredDocument]:
        return await self._documents.list_all()

    async def delete(self, document_id: uuid.UUID) -> None:
        if not await self._documents.delete(document_id):
            raise AppError("not_found", "Документ не найден", status_code=404)
        await self._index.delete_documents([document_id])
        await self._documents.commit()
