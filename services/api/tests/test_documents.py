import hashlib
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.security import API_KEY_HEADER
from app.services import ingestion
from tests.factories import TEST_ADMIN_API_KEY, InMemoryDocumentRepository, build_app

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF"
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
) -> httpx.Response:
    response: httpx.Response = client.post(
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
    assert body["chunks"] == 0
    stored = repository.documents[hashlib.sha256(PDF).hexdigest()]
    assert str(stored.id) == body["id"]
    assert (stored.filename, stored.mime) == ("report.pdf", "application/pdf")


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
    assert second.json() == {"id": first.json()["id"], "chunks": 0, "status": "exists"}


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
