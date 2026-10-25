from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.adapters.embedders.base import ChunkTooLargeError
from app.core.config import DatabaseSettings, Settings
from tests.factories import TEST_ADMIN_API_KEY, TEST_DATABASE_URL, TEST_QDRANT_URL, build_app


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("QDRANT_URL", TEST_QDRANT_URL)
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_API_KEY)


def test_defaults() -> None:
    assert Settings().llm_provider == "stub"


def test_reads_env_variable() -> None:
    assert str(Settings().database_url) == TEST_DATABASE_URL


def test_reads_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATABASE_URL")
    (tmp_path / ".env").write_text(f"DATABASE_URL={TEST_DATABASE_URL}\n", encoding="utf-8")

    assert str(Settings().database_url) == TEST_DATABASE_URL


def test_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")

    with pytest.raises(ValidationError, match="database_url"):
        Settings()


def test_rejects_non_postgres_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@localhost/db")

    with pytest.raises(ValidationError, match="database_url"):
        Settings()


def test_requires_qdrant_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL")

    with pytest.raises(ValidationError, match="qdrant_url"):
        Settings()


def test_requires_admin_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_API_KEY")

    with pytest.raises(ValidationError, match="admin_api_key"):
        Settings()


def test_admin_api_key_is_hidden_in_repr() -> None:
    assert TEST_ADMIN_API_KEY not in repr(Settings())


def test_chunk_defaults() -> None:
    settings = Settings()

    assert (settings.chunk_size, settings.chunk_overlap) == (300, 45)


def test_chunk_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings()


def test_app_refuses_chunk_larger_than_embedder_limit() -> None:
    settings = Settings(chunk_size=500)

    with (
        pytest.raises(ChunkTooLargeError, match="chunk_size=500"),
        TestClient(build_app(settings=settings)),
    ):
        pass


def test_openai_embedder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDER", "openai")

    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings()


def test_openai_key_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    assert "sk-secret" not in repr(Settings())


def test_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValidationError, match="llm_provider"):
        Settings()


def test_app_exposes_settings_from_lifespan() -> None:
    settings = Settings()
    app = build_app(settings=settings)

    with TestClient(app):
        assert app.state.settings is settings


def test_database_settings_do_not_require_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL")

    assert str(DatabaseSettings().database_url) == TEST_DATABASE_URL
