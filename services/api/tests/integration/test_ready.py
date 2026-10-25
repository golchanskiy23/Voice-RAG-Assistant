from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from testcontainers.core.container import DockerContainer

from app.api.routes import health
from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


@contextmanager
def paused(container: DockerContainer) -> Iterator[None]:
    wrapped = container.get_wrapped_container()
    wrapped.pause()
    try:
        yield
    finally:
        wrapped.unpause()


def get_ready(settings: Settings) -> tuple[int, dict[str, bool]]:
    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/ready")
    return response.status_code, response.json()["checks"]


def test_ready_with_real_dependencies(settings: Settings) -> None:
    status, checks = get_ready(settings)

    assert status == 200
    assert checks == {"postgres": True, "qdrant": True, "llm": True}


@pytest.mark.parametrize("resource", ["postgres", "qdrant"])
def test_ready_reports_paused_dependency(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    resource: str,
) -> None:
    monkeypatch.setattr(health, "READY_TIMEOUT_SECONDS", 0.5)
    container: DockerContainer = request.getfixturevalue(resource)

    with paused(container):
        status, checks = get_ready(settings)

    assert status == 503
    assert checks[resource] is False
    assert all(ok for name, ok in checks.items() if name != resource)


def test_ready_recovers_after_dependency_returns(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, postgres: DockerContainer
) -> None:
    monkeypatch.setattr(health, "READY_TIMEOUT_SECONDS", 0.5)

    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/ready").status_code == 200
        with paused(postgres):
            assert client.get("/ready").status_code == 503
        assert client.get("/ready").status_code == 200
