import asyncio

import pytest
from fastapi.testclient import TestClient

from app.adapters.llm.stub import StubLLMClient
from app.api.routes import health
from tests.factories import FakeDatabase, FakeQdrant, build_app


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_llm_available(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": True, "qdrant": True, "embedder": True, "llm": True},
    }


class UnavailableLLM(StubLLMClient):
    async def is_ready(self) -> bool:
        return False


def test_ready_when_llm_unavailable() -> None:
    with TestClient(build_app(llm_factory=UnavailableLLM)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"postgres": True, "qdrant": True, "embedder": True, "llm": False},
    }


def test_lifespan_creates_single_client_and_closes_it() -> None:
    app = build_app()
    with TestClient(app) as client:
        llm = app.state.llm
        client.get("/ready")
        client.post("/ask", json={"question": "Что такое RAG?"})
        assert app.state.llm is llm

    assert llm._closed is True


class SlowLLM(StubLLMClient):
    async def is_ready(self) -> bool:
        await asyncio.sleep(1)
        return True


class BrokenLLM(StubLLMClient):
    async def is_ready(self) -> bool:
        raise ConnectionError("unreachable")


@pytest.mark.parametrize("llm_factory", [SlowLLM, BrokenLLM])
def test_ready_treats_timeout_and_errors_as_not_ready(
    monkeypatch: pytest.MonkeyPatch, llm_factory: type[StubLLMClient]
) -> None:
    monkeypatch.setattr(health, "READY_TIMEOUT_SECONDS", 0.05)

    with TestClient(build_app(llm_factory=llm_factory)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "postgres": True,
        "qdrant": True,
        "embedder": True,
        "llm": False,
    }


def test_ready_reports_postgres_unavailable() -> None:
    with TestClient(build_app(database=FakeDatabase(ready=False))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "postgres": False,
        "qdrant": True,
        "embedder": True,
        "llm": True,
    }


def test_lifespan_closes_database() -> None:
    database = FakeDatabase()
    with TestClient(build_app(database=database)):
        assert database.closed is False

    assert database.closed is True


def test_ready_reports_qdrant_unavailable() -> None:
    with TestClient(build_app(qdrant=FakeQdrant(ready=False))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "postgres": True,
        "qdrant": False,
        "embedder": True,
        "llm": True,
    }


def test_lifespan_closes_qdrant() -> None:
    qdrant = FakeQdrant()
    with TestClient(build_app(qdrant=qdrant)):
        assert qdrant.closed is False

    assert qdrant.closed is True
