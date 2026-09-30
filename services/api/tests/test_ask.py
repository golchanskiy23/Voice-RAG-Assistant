import json

from fastapi.testclient import TestClient

from app.adapters.llm.base import Completion, LLMRequest
from app.adapters.llm.stub import StubLLMClient
from app.services.llm_answer import NO_ANSWER
from tests.factories import build_app
from tests.test_search import FAQ, GUIDE, upload


def test_ask_without_documents_refuses(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Что такое RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_ANSWER
    assert body["sources"] == []
    assert body["confidence"] == 0.0
    assert {"embed", "search", "total"} <= set(body["latency"])


def test_ask_answers_with_source_file_and_page(client: TestClient) -> None:
    upload(client, "guide.md", GUIDE)
    upload(client, "faq.md", FAQ)

    response = client.post("/ask", json={"question": "Какие файлы отклоняются по лимиту?"})

    body = response.json()
    assert body["answer"] == "Файлы больше двадцати мегабайт отклоняются."
    [source] = body["sources"]
    assert source == {
        "filename": "faq.md",
        "page": None,
        "section": "FAQ > Лимиты",
        "quote": "Файлы больше двадцати мегабайт отклоняются.",
    }
    assert body["confidence"] == 0.5
    assert {"embed", "search", "llm", "total"} <= set(body["latency"])


class EchoLLM(StubLLMClient):
    async def complete(self, request: LLMRequest) -> Completion:
        question = request.messages[-1].content.rsplit("Вопрос: ", 1)[-1]
        answer = {"answer": f"echo: {question}", "sources": [1], "confidence": 0.5}
        return Completion(json.dumps(answer, ensure_ascii=False), self.name)


def test_ask_uses_llm_from_lifespan() -> None:
    with TestClient(build_app(llm_factory=EchoLLM)) as client:
        upload(client, "faq.md", FAQ)
        response = client.post("/ask", json={"question": "привет"})

    body = response.json()
    assert body["answer"] == "echo: привет"
    assert body["sources"][0]["filename"] == "faq.md"


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "question" in error["message"]
