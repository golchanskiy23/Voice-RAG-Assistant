import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from app.adapters.embedders.base import Embedder
from app.core.errors import AppError
from app.parsing.markdown import parse_markdown
from app.parsing.models import DocumentParseError, ParsedDocument
from app.parsing.pdf import parse_pdf
from app.repositories.documents import (
    DocumentRepository,
    NewChunk,
    NewDocument,
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


class IngestionService:
    def __init__(self, documents: DocumentRepository, embedder: Embedder, chunker: Chunker) -> None:
        self._documents = documents
        self._embedder = embedder
        self._chunker = chunker

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
        stored, status = await self._documents.save(document, chunks)
        return UploadResult(stored, status)
