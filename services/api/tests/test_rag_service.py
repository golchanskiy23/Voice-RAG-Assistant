import asyncio

from app.adapters.llm.stub import StubLLMClient
from app.services.llm_answer import NO_ANSWER
from app.services.rag import RAGService


class ScriptedLLM(StubLLMClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = responses
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


VALID = '{"answer": "Готово", "confidence": 0.7}'
OUT_OF_CONTEXT = '{"answer": "Готово", "sources": [2], "confidence": 0.7}'


def test_valid_answer_needs_single_call() -> None:
    llm = ScriptedLLM([VALID])

    answer = asyncio.run(RAGService(llm).answer("вопрос"))

    assert answer.text == "Готово"
    assert answer.confidence == 0.7
    assert len(llm.prompts) == 1


def test_invalid_answer_is_retried_with_error_details() -> None:
    llm = ScriptedLLM([OUT_OF_CONTEXT, VALID])

    answer = asyncio.run(RAGService(llm).answer("вопрос"))

    assert answer.text == "Готово"
    assert len(llm.prompts) == 2
    assert "вне контекста" in llm.prompts[1]
    assert OUT_OF_CONTEXT in llm.prompts[1]


def test_falls_back_after_second_invalid_answer() -> None:
    llm = ScriptedLLM(["не JSON", OUT_OF_CONTEXT])

    answer = asyncio.run(RAGService(llm).answer("вопрос"))

    assert answer.text == NO_ANSWER
    assert answer.confidence == 0.0
    assert len(llm.prompts) == 2
