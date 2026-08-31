import uuid
from collections.abc import Callable

from fastapi import FastAPI

from app.adapters.embedders.base import Embedder
from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.api.deps import get_document_repository
from app.core.config import Settings
from app.core.database import Database
from app.core.qdrant import Qdrant
from app.main import create_app
from app.repositories.documents import DocumentRepository, NewDocument, StoredDocument

TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:1/test"
TEST_QDRANT_URL = "http://localhost:1"
TEST_ADMIN_API_KEY = "test-admin-key"


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
        super().__init__(TEST_QDRANT_URL, check_compatibility=False)
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

    async def get_or_create(self, document: NewDocument) -> tuple[StoredDocument, bool]:
        if document.sha256 in self.documents:
            return self.documents[document.sha256], False
        stored = StoredDocument(
            id=uuid.uuid4(), filename=document.filename, sha256=document.sha256, mime=document.mime
        )
        self.documents[document.sha256] = stored
        return stored, True


def build_app(
    llm_factory: Callable[[], LLMClient] = StubLLMClient,
    database: FakeDatabase | None = None,
    qdrant: FakeQdrant | None = None,
    settings: Settings | None = None,
    documents: DocumentRepository | None = None,
    embedder_factory: Callable[[], Embedder] = HashingEmbedder,
) -> FastAPI:
    fake_database = database or FakeDatabase()
    fake_qdrant = qdrant or FakeQdrant()
    repository = documents or InMemoryDocumentRepository()
    app = create_app(
        settings=settings or make_settings(),
        llm_factory=llm_factory,
        database_factory=lambda _: fake_database,
        qdrant_factory=lambda _: fake_qdrant,
        embedder_factory=embedder_factory,
    )
    app.dependency_overrides[get_document_repository] = lambda: repository
    return app
