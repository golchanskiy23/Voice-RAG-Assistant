import uuid
from collections.abc import Callable, Sequence

from fastapi import FastAPI

from app.adapters.embedders.base import Embedder
from app.adapters.embedders.hashing import HashingEmbedder, words
from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.adapters.stt.base import SpeechToText
from app.adapters.stt.stub import StubSTT
from app.adapters.tts.stub import StubTTS
from app.adapters.vector_stores.base import SearchHit
from app.api.deps import get_document_repository
from app.core.config import Settings
from app.core.database import Database
from app.core.qdrant import Qdrant
from app.main import create_app
from app.repositories.documents import (
    DocumentRepository,
    NewChunk,
    NewDocument,
    SavedChunk,
    SaveResult,
    StoredDocument,
)

TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:1/test"
TEST_QDRANT_URL = "http://localhost:1"
TEST_ADMIN_API_KEY = "test-admin-key"


def count_words(text: str) -> int:
    return len(text.split())


def make_settings() -> Settings:
    return Settings(
        database_url=TEST_DATABASE_URL,
        qdrant_url=TEST_QDRANT_URL,
        admin_api_key=TEST_ADMIN_API_KEY,
    )


class FakeDatabase(Database):
    def __init__(self, ready: bool = True) -> None:
        super().__init__(TEST_DATABASE_URL)
        self.ready = ready
        self.closed = False

    async def is_ready(self) -> bool:
        return self.ready

    async def close(self) -> None:
        self.closed = True
        await super().close()


class FakeQdrant(Qdrant):
    def __init__(self, ready: bool = True) -> None:
        super().__init__(location=":memory:")
        self.ready = ready
        self.closed = False

    async def is_ready(self) -> bool:
        return self.ready

    async def close(self) -> None:
        self.closed = True
        await super().close()


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.documents: dict[str, StoredDocument] = {}
        self.chunks: dict[uuid.UUID, list[SavedChunk]] = {}
        self._pending: tuple[StoredDocument, list[SavedChunk], list[StoredDocument]] | None = None
        self.commits = 0
        self.rollbacks = 0

    async def find_by_sha256(self, sha256: str) -> StoredDocument | None:
        return self.documents.get(sha256)

    async def save(self, document: NewDocument, chunks: Sequence[NewChunk]) -> SaveResult:
        if document.sha256 in self.documents:
            return SaveResult(self.documents[document.sha256], "exists", [], [])
        old = [d for d in self.documents.values() if d.filename == document.filename]
        stored = StoredDocument(
            id=uuid.uuid4(),
            filename=document.filename,
            sha256=document.sha256,
            mime=document.mime,
            pages=document.pages,
            chunks=len(chunks),
        )
        saved = [SavedChunk(id=uuid.uuid4(), chunk=chunk) for chunk in chunks]
        self._pending = (stored, saved, old)
        return SaveResult(stored, "replaced" if old else "created", saved, [d.id for d in old])

    async def list_all(self) -> list[StoredDocument]:
        return list(self.documents.values())

    async def delete(self, document_id: uuid.UUID) -> bool:
        for sha256, document in list(self.documents.items()):
            if document.id == document_id:
                del self.documents[sha256]
                self.chunks.pop(document_id, None)
                return True
        return False

    async def commit(self) -> None:
        if self._pending is not None:
            stored, saved, old = self._pending
            for stale in old:
                del self.documents[stale.sha256]
                self.chunks.pop(stale.id, None)
            self.documents[stored.sha256] = stored
            self.chunks[stored.id] = saved
            self._pending = None
        self.commits += 1

    async def rollback(self) -> None:
        self._pending = None
        self.rollbacks += 1


class InMemoryKeywordIndex:
    def __init__(self, repository: InMemoryDocumentRepository) -> None:
        self._repository = repository

    async def search(
        self, query: str, k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        terms = set(words(query))
        hits = []
        for document in self._repository.documents.values():
            if document_ids and document.id not in document_ids:
                continue
            for saved in self._repository.chunks.get(document.id, []):
                overlap = len(terms & set(words(saved.chunk.text)))
                if overlap:
                    hits.append(
                        SearchHit(
                            chunk_id=saved.id,
                            document_id=document.id,
                            filename=document.filename,
                            ord=saved.chunk.ord,
                            text=saved.chunk.text,
                            page=saved.chunk.page,
                            section=saved.chunk.section,
                            score=float(overlap),
                            found_by=("bm25",),
                        )
                    )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:k]


def build_app(
    llm_factory: Callable[[], LLMClient] = StubLLMClient,
    database: FakeDatabase | None = None,
    qdrant: FakeQdrant | None = None,
    settings: Settings | None = None,
    documents: DocumentRepository | None = None,
    embedder_factory: Callable[[], Embedder] = HashingEmbedder,
    count_tokens: Callable[[str], int] = count_words,
    stt_factory: Callable[[], SpeechToText] = StubSTT,
) -> FastAPI:
    fake_database = database or FakeDatabase()
    fake_qdrant = qdrant or FakeQdrant()
    repository = documents or InMemoryDocumentRepository()
    keyword = InMemoryKeywordIndex(
        repository
        if isinstance(repository, InMemoryDocumentRepository)
        else InMemoryDocumentRepository()
    )
    app = create_app(
        settings=settings or make_settings(),
        llm_factory=llm_factory,
        database_factory=lambda _: fake_database,
        qdrant_factory=lambda _: fake_qdrant,
        embedder_factory=embedder_factory,
        count_tokens=count_tokens,
        keyword_factory=lambda _: keyword,
        stt_factory=stt_factory,
        tts_factory=StubTTS,
    )
    app.dependency_overrides[get_document_repository] = lambda: repository
    return app
