import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import AppError
from app.core.request_id import REQUEST_ID_HEADER


@pytest.fixture
def app(app: FastAPI) -> FastAPI:
    @app.get("/raise/app-error")
    async def raise_app_error() -> None:
        raise AppError("document_not_found", "Документ не найден", status_code=404)

    @app.get("/raise/unexpected")
    async def raise_unexpected() -> None:
        raise RuntimeError("boom")

    return app


def test_app_error_format(client: TestClient) -> None:
    response = client.get("/raise/app-error", headers={REQUEST_ID_HEADER: "req-1"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "document_not_found",
            "message": "Документ не найден",
            "request_id": "req-1",
        }
    }


def test_unknown_route_uses_error_format(client: TestClient) -> None:
    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_method_not_allowed_uses_error_format(client: TestClient) -> None:
    response = client.get("/ask")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_unexpected_error_hides_details(client: TestClient) -> None:
    response = client.get("/raise/unexpected", headers={REQUEST_ID_HEADER: "req-2"})

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "internal_error"
    assert "boom" not in error["message"]
    assert error["request_id"] == "req-2"


def test_request_id_is_generated_and_returned(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc"})

    assert response.headers[REQUEST_ID_HEADER] == "abc"
