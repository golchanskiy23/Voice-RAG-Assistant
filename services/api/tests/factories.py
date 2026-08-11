from collections.abc import Callable

from fastapi import FastAPI

from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.core.config import Settings
from app.core.database import Database
from app.core.qdrant import Qdrant
from app.main import create_app

TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:1/test"
TEST_QDRANT_URL = "http://localhost:1"


def make_settings() -> Settings:
    return Settings(database_url=TEST_DATABASE_URL, qdrant_url=TEST_QDRANT_URL)


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


def build_app(
    llm_factory: Callable[[], LLMClient] = StubLLMClient,
    database: FakeDatabase | None = None,
    qdrant: FakeQdrant | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    fake_database = database or FakeDatabase()
    fake_qdrant = qdrant or FakeQdrant()
    return create_app(
        settings=settings or make_settings(),
        llm_factory=llm_factory,
        database_factory=lambda _: fake_database,
        qdrant_factory=lambda _: fake_qdrant,
    )
