import uuid
from collections.abc import AsyncIterator, Sequence

from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.llm.base import (
    Completion,
    LLMChunk,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.adapters.llm.stub import StubLLMClient
from app.adapters.vector_stores.base import SearchHit, StoreName, VectorPoint
from app.core.config import PROMPTS_DIR
from app.services.prompts import load_answer_prompt
from app.services.rag import RAGService, RAGSettings
from app.services.search import SearchService
from tests.factories import count_words

PROMPT = load_answer_prompt(PROMPTS_DIR, "v1")


def hit(
    text: str, filename: str = "guide.md", page: int | None = 1, score: float = 0.9
) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        ord=0,
        text=text,
        page=page,
        section="Раздел",
        score=score,
    )


class FixedStore:
    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self._hits = list(hits)

    @property
    def name(self) -> StoreName:
        return "qdrant"

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        return None

    async def search(
        self, vector: list[float], k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        return self._hits[:k]

    async def delete_documents(self, document_ids: Sequence[uuid.UUID]) -> None:
        return None


class ScriptedLLM(StubLLMClient):
    def __init__(self, responses: list[str], fail_stream_after: int | None = None) -> None:
        super().__init__()
        self._responses = responses
        self._fail_stream_after = fail_stream_after
        self.requests: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, request: LLMRequest) -> Completion:
        self.requests.append(request)
        text = self._responses[len(self.requests) - 1]
        return Completion(text=text, provider=self.name, usage=Usage(100, 20))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        self.requests.append(request)
        text = self._responses[len(self.requests) - 1]
        for i in range(0, len(text), 3):
            if self._fail_stream_after is not None and i >= self._fail_stream_after:
                raise LLMError(self.name, "stream broken")
            yield TextDelta(text[i : i + 3])
        yield StreamEnd(provider=self.name, usage=Usage(100, 20))


def make_search(hits: Sequence[SearchHit]) -> SearchService:
    return SearchService(HashingEmbedder(), {"qdrant": FixedStore(hits)})


def make_service(
    llm: ScriptedLLM | StubLLMClient,
    hits: Sequence[SearchHit],
    settings: RAGSettings | None = None,
) -> RAGService:
    return RAGService(make_search(hits), llm, PROMPT, count_words, settings)
