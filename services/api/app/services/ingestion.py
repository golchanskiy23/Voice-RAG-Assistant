import hashlib
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from app.core.errors import AppError
from app.repositories.documents import DocumentRepository, NewDocument, StoredDocument

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
    created: bool


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


class IngestionService:
    def __init__(self, documents: DocumentRepository) -> None:
        self._documents = documents

    async def upload(self, filename: str, content: bytes) -> UploadResult:
        if not content:
            raise AppError("empty_file", "Файл пустой", status_code=400)
        mime = detect_mime(filename, content)
        sha256 = hashlib.sha256(content).hexdigest()
        document, created = await self._documents.get_or_create(
            NewDocument(filename=PurePath(filename).name, sha256=sha256, mime=mime)
        )
        return UploadResult(document=document, created=created)
