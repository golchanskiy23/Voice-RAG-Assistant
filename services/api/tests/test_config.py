from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert Settings().llm_provider == "stub"


def test_reads_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "stub")

    assert Settings().llm_provider == "stub"


def test_reads_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    (tmp_path / ".env").write_text("LLM_PROVIDER=unknown\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="llm_provider"):
        Settings()


def test_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValidationError, match="llm_provider"):
        Settings()


def test_app_exposes_settings_from_lifespan() -> None:
    settings = Settings(llm_provider="stub")
    app = create_app(settings=settings)

    with TestClient(app):
        assert app.state.settings is settings
