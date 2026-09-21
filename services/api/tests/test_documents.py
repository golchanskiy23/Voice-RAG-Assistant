import hashlib
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.adapters.embedders.base import EMBEDDING_DIM
from app.api.security import API_KEY_HEADER
from app.services import ingestion
from tests.factories import TEST_ADMIN_API_KEY, InMemoryDocumentRepository, build_app

FIXTURES = Path(__file__).parent / "fixtures"
PDF = (FIXTURES / "regulation_ru.pdf").read_bytes()
MARKDOWN = "# Заголовок\n\nТекст документа.".encode()
AUTH = {API_KEY_HEADER: TEST_ADMIN_API_KEY}


@pytest.fixture
def repository() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@pytest.fixture
def client(repository: InMemoryDocumentRepository) -> Iterator[TestClient]:
    with TestClient(build_app(documents=repository)) as test_client:
        yield test_client


def upload(
    client: TestClient, filename: str, content: bytes, headers: dict[str, str] = AUTH
) -> httpx2.Response:
    response: httpx2.Response = client.post(
        "/documents", files={"file": (filename, content)}, headers=headers
    )
    return response


def test_upload_pdf_creates_document(
    client: TestClient, repository: InMemoryDocumentRepository
) -> None:
    response = upload(client, "report.pdf", PDF)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "created"
    assert body["pages"] == 2
    stored = repository.documents[hashlib.sha256(PDF).hexdigest()]
    chunks = repository.chunks[stored.id]
    assert str(stored.id) == body["id"]
    assert (stored.filename, stored.mime) == ("report.pdf", "application/pdf")
    assert body["chunks"] == len(chunks) > 0
    assert all(len(c.chunk.embedding) == EMBEDDING_DIM for c in chunks)
    assert all(c.chunk.section and c.chunk.page in (1, 2) for c in chunks)
    assert [c.chunk.ord for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("filename", ["notes.md", "NOTES.MARKDOWN"])
def test_upload_markdown(
    client: TestClient, repository: InMemoryDocumentRepository, filename: str
) -> None:
    response = upload(client, filename, MARKDOWN)

    assert response.status_code == 201
    assert next(iter(repository.documents.values())).mime == "text/markdown"


def test_same_content_returns_existing_document(client: TestClient) -> None:
    first = upload(client, "a.md", MARKDOWN)
    second = upload(client, "renamed.md", MARKDOWN)

    assert second.status_code == 200
    assert second.json() == {
        "id": first.json()["id"],
        "chunks": first.json()["chunks"],
        "pages": None,
        "status": "exists",
    }


def test_new_version_with_same_name_replaces_document(
    client: TestClient, repository: InMemoryDocumentRepository
) -> None:
    first = upload(client, "guide.md", MARKDOWN)
    second = upload(client, "guide.md", "# Заголовок\n\nНовая редакция текста.".encode())

    assert second.status_code == 201
    assert second.json()["status"] == "replaced"
    assert second.json()["id"] != first.json()["id"]
    assert [d.id for d in repository.documents.values()] == [uuid.UUID(second.json()["id"])]


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("scan.pdf", (FIXTURES / "scan_without_text.pdf").read_bytes(), "no_text"),
        ("broken.pdf", b"%PDF-1.7\nnot really a pdf", "invalid_pdf"),
        ("headings.md", b"# Only heading\n", "no_text"),
    ],
)
def test_unprocessable_documents(
    client: TestClient,
    repository: InMemoryDocumentRepository,
    filename: str,
    content: bytes,
    code: str,
) -> None:
    response = upload(client, filename, content)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert repository.documents == {}


def test_filename_path_is_stripped(
    client: TestClient, repository: InMemoryDocumentRepository
) -> None:
    upload(client, "../../etc/notes.md", MARKDOWN)

    assert next(iter(repository.documents.values())).filename == "notes.md"


@pytest.mark.parametrize(
    "headers",
    [{}, {API_KEY_HEADER: "wrong"}, {API_KEY_HEADER: ""}],
    ids=["missing", "wrong", "empty"],
)
def test_requires_admin_key(
    client: TestClient, repository: InMemoryDocumentRepository, headers: dict[str, str]
) -> None:
    response = upload(client, "a.md", MARKDOWN, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert repository.documents == {}


@pytest.mark.parametrize(
    ("filename", "content", "fragment"),
    [
        ("notes.txt", b"text", "Поддерживаются только PDF и Markdown"),
        ("noextension", b"text", "Поддерживаются только PDF и Markdown"),
        ("fake.pdf", b"not a pdf", "не является PDF"),
        ("cp1251.md", "Привет".encode("cp1251"), "UTF-8"),
    ],
)
def test_rejects_unsupported_files(
    client: TestClient, filename: str, content: bytes, fragment: str
) -> None:
    response = upload(client, filename, content)

    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "unsupported_media_type"
    assert fragment in error["message"]


def test_rejects_empty_file(client: TestClient) -> None:
    response = upload(client, "empty.md", b"")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_rejects_too_large_file(
    client: TestClient, repository: InMemoryDocumentRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingestion, "MAX_UPLOAD_BYTES", 1024)

    response = upload(client, "big.md", b"a" * 1025)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert repository.documents == {}


def test_file_at_size_limit_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingestion, "MAX_UPLOAD_BYTES", 1024)

    assert upload(client, "limit.md", b"a" * 1024).status_code == 201


def test_missing_file_is_validation_error(client: TestClient) -> None:
    response = client.post("/documents", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
