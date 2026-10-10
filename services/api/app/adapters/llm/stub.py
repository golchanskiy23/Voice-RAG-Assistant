import asyncio
import json
import re
from collections.abc import AsyncIterator

from app.adapters.llm.base import Completion, LLMChunk, LLMRequest, StreamEnd, TextDelta
from app.services.llm_answer import NO_ANSWER

FIRST_CHUNK = re.compile(
    r'<фрагмент номер="1"[^>]*>\n(?P<xml>.*?)\n</фрагмент>'
    r"|^\[1\] \(.*?\)\n(?P<plain>.+?)(?:\n\n\[2\] |\n\nВопрос:)",
    re.S | re.M,
)
SENTENCE = re.compile(r".+?[.!?](?=\s|$)", re.S)


def extractive_answer(request: LLMRequest) -> str:
    user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    match = FIRST_CHUNK.search(user)
    if match is None:
        answer = {"answer": NO_ANSWER, "sources": [], "confidence": 0.0}
    else:
        text = " ".join((match["xml"] or match["plain"]).split())
        sentence = SENTENCE.match(text)
        answer = {
            "answer": (sentence[0] if sentence else text)[:300],
            "sources": [1],
            "confidence": 0.5,
        }
    return json.dumps(answer, ensure_ascii=False)


class StubLLMClient:
    def __init__(self, token_delay: float = 0.0) -> None:
        self._token_delay = token_delay
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    async def complete(self, request: LLMRequest) -> Completion:
        return Completion(text=extractive_answer(request), provider=self.name)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        raw = extractive_answer(request)
        for token in re.findall(r"\S+\s*|\s+", raw):
            await asyncio.sleep(self._token_delay)
            yield TextDelta(token)
        yield StreamEnd(provider=self.name)

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
