from fastapi.testclient import TestClient

from app.adapters.llm.stub import StubLLMClient
from app.main import create_app


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_llm_available(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"llm": True}}


class UnavailableLLM(StubLLMClient):
    async def is_ready(self) -> bool:
        return False


def test_ready_when_llm_unavailable() -> None:
    with TestClient(create_app(llm_factory=UnavailableLLM)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "checks": {"llm": False}}


def test_lifespan_creates_single_client_and_closes_it() -> None:
    app = create_app()
    with TestClient(app) as client:
        llm = app.state.llm
        client.get("/ready")
        client.post("/ask", json={"question": "Что такое RAG?"})
        assert app.state.llm is llm

    assert llm._closed is True
