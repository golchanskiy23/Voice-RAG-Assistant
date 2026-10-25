from fastapi.testclient import TestClient

from app.adapters.llm.stub import StubLLMClient
from app.main import create_app
from app.services.llm_answer import NO_ANSWER, LLMAnswer


def test_ask_returns_stub_answer(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Что такое RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []
    assert body["confidence"] == 0.0
    assert "llm" in body["latency"]


class EchoLLM(StubLLMClient):
    async def complete(self, prompt: str) -> str:
        return LLMAnswer(answer=f"echo: {prompt}", confidence=0.5).model_dump_json()


def test_ask_uses_llm_from_lifespan() -> None:
    with TestClient(create_app(llm_factory=EchoLLM)) as client:
        response = client.post("/ask", json={"question": "привет"})

    body = response.json()
    assert body["answer"] == "echo: привет"
    assert body["confidence"] == 0.5


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "question" in error["message"]
