from collections.abc import Iterator
from pathlib import Path

import docker
import pytest
from alembic import command
from alembic.config import Config
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.qdrant import QdrantContainer

from app.core.config import Settings
from tests.factories import TEST_ADMIN_API_KEY

POSTGRES_IMAGE = "pgvector/pgvector:0.8.0-pg17"
QDRANT_IMAGE = "qdrant/qdrant:v1.20.0"
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def docker_available() -> bool:
    try:
        docker.from_env().ping()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def require_docker() -> None:
    if not docker_available():
        pytest.skip("Docker недоступен")


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def qdrant() -> Iterator[QdrantContainer]:
    with QdrantContainer(QDRANT_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def database_url(postgres: PostgresContainer) -> str:
    return postgres.get_connection_url()


@pytest.fixture(scope="session")
def qdrant_url(qdrant: QdrantContainer) -> str:
    return f"http://{qdrant.rest_host_address}"


def run_alembic(database_url: str, action: str, revision: str) -> None:
    config = Config(str(ALEMBIC_INI))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", database_url)
        getattr(command, action)(config, revision)


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    run_alembic(database_url, "upgrade", "head")
    return database_url


@pytest.fixture
def settings(migrated_database_url: str, qdrant_url: str) -> Settings:
    return Settings(
        database_url=migrated_database_url,
        qdrant_url=qdrant_url,
        admin_api_key=TEST_ADMIN_API_KEY,
    )
