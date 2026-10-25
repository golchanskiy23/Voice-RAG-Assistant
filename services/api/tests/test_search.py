from collections.abc import Iterator, Sequence

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.adapters.vector_stores.base import VectorPoint
from app.api.security import API_KEY_HEADER
from app.core.config import Settings
from tests.factories import (
    TEST_ADMIN_API_KEY,
    TEST_DATABASE_URL,
    TEST_QDRANT_URL,
    InMemoryDocumentRepository,
    build_app,
)

GUIDE = (
    "# Руководство\n\n## Загрузка\n\nДокументы загружаются администратором через API.\n\n"
    "## Голос\n\nОтвет озвучивается по предложениям, чтобы звук начинался раньше.\n"
).encode()
FAQ = "# FAQ\n\n## Лимиты\n\nФайлы больше двадцати мегабайт отклоняются.\n".encode()


def settings(store: str = "qdrant") -> Settings:
    return Settings(
        database_url=TEST_DATABASE_URL,
        qdrant_url=TEST_QDRANT_URL,
        admin_api_key=TEST_ADMIN_API_KEY,
        vector_store=store,
    )


@pytest.fixture
def repository() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@pytest.fixture
def client(repository: InMemoryDocumentRepository) -> Iterator[TestClient]:
    with TestClient(build_app(documents=repository, settings=settings())) as test_client:
        yield test_client


def upload(client: TestClient, filename: str, content: bytes) -> str:
    response: httpx2.Response = client.post(
        "/documents",
        files={"file": (filename, content)},
        headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
    )
    assert response.status_code == 201
    document_id: str = response.json()["id"]
    return document_id


def search(client: TestClient, **payload: object) -> httpx2.Response:
    response: httpx2.Response = client.post("/search", json=payload)
    return response


def test_uploaded_chunks_are_searchable(client: TestClient) -> None:
    upload(client, "guide.md", GUIDE)

    response = search(client, query="озвучивается ответ по предложениям", k=2)

    assert response.status_code == 200
    body = response.json()
    assert body["store"] == "qdrant"
    top = body["chunks"][0]
    assert top["section"] == "Руководство > Голос"
    assert top["filename"] == "guide.md"
    assert top["source"] == "vector"
    assert body["chunks"][0]["score"] >= body["chunks"][1]["score"]
    assert set(body["latency"]) == {"embed", "search"}


def test_k_limits_results(client: TestClient) -> None:
    upload(client, "guide.md", GUIDE)

    assert len(search(client, query="документы", k=1).json()["chunks"]) == 1


def test_filter_by_document(client: TestClient) -> None:
    upload(client, "guide.md", GUIDE)
    faq_id = upload(client, "faq.md", FAQ)

    chunks = search(client, query="документы загружаются", k=5, doc_ids=[faq_id]).json()["chunks"]

    assert chunks and {c["document_id"] for c in chunks} == {faq_id}


def test_replaced_document_disappears_from_index(client: TestClient) -> None:
    old_id = upload(client, "guide.md", GUIDE)
    new_id = upload(client, "guide.md", "# Руководство\n\nНовая редакция без голоса.\n".encode())

    ids = {c["document_id"] for c in search(client, query="голос", k=10).json()["chunks"]}

    assert ids == {new_id}
    assert old_id not in ids


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "q", "k": 0},
        {"query": "q", "k": 51},
        {"query": "q", "store": "faiss"},
    ],
)
def test_invalid_requests(client: TestClient, payload: dict[str, object]) -> None:
    response = search(client, **payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


class FailingIndex:
    name = "qdrant"

    def __init__(self) -> None:
        self.deleted: list[object] = []

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        raise ConnectionError("qdrant недоступен")

    async def delete_documents(self, document_ids: Sequence[object]) -> None:
        self.deleted.extend(document_ids)


def test_indexing_failure_rolls_back_document(repository: InMemoryDocumentRepository) -> None:
    app = build_app(documents=repository, settings=settings())
    index = FailingIndex()
    with TestClient(app) as client:
        app.state.index = index
        response = client.post(
            "/documents",
            files={"file": ("guide.md", GUIDE)},
            headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "indexing_failed"
    assert repository.documents == {}
    assert (repository.commits, repository.rollbacks) == (0, 1)
    assert len(index.deleted) == 1
