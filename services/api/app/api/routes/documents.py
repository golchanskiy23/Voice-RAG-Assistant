import uuid

from fastapi import APIRouter, Response, UploadFile, status

from app.api.deps import IngestionServiceDep
from app.api.schemas import DocumentListResponse, DocumentOut, DocumentUploadResponse
from app.api.security import AdminKey
from app.core.errors import AppError
from app.services import ingestion

READ_CHUNK_BYTES = 1024 * 1024

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[AdminKey])


async def read_limited(file: UploadFile, limit: int) -> bytes:
    parts: list[bytes] = []
    size = 0
    while chunk := await file.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            raise AppError(
                "payload_too_large",
                f"Файл больше {limit // (1024 * 1024)} МБ",
                status_code=413,
            )
        parts.append(chunk)
    return b"".join(parts)


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, service: IngestionServiceDep, response: Response
) -> DocumentUploadResponse:
    content = await read_limited(file, ingestion.MAX_UPLOAD_BYTES)
    result = await service.upload(file.filename or "", content)
    if result.status == "exists":
        response.status_code = status.HTTP_200_OK
    return DocumentUploadResponse(
        id=result.document.id,
        chunks=result.document.chunks,
        pages=result.document.pages,
        status=result.status,
    )


@router.get("")
async def list_documents(service: IngestionServiceDep) -> DocumentListResponse:
    documents = await service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentOut(id=d.id, filename=d.filename, mime=d.mime, pages=d.pages, chunks=d.chunks)
            for d in documents
        ]
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, service: IngestionServiceDep) -> Response:
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
